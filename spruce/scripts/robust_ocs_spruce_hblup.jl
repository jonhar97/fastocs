# =============================================================================
# HBLUP VERSION (2026-09-21) -- changes relative to robust_ocs_spruce_4.jl:
#   * g_vec = single-trait Hjd17 AI-REML HBLUP EBVs (load_spruce_hblup.jl),
#     all 5525 trees kept as candidates, GAMMA re-calibrated to 400.
#   * Ω = the AI-REML prediction error covariance matrix (PEV), replacing the
#     MCMC posterior-draw factor. PEV is NOT low rank (rank 100 holds only
#     ~10% of its trace), so it is represented like G: a rank-RANK_OMEGA RSVD
#     factor plus an exact diagonal correction,
#         Ω ≈ FΩ FΩ' + diag(dΩ),   dΩ = diag(PEV) - rowsums(FΩ.^2) >= 0,
#     which reproduces sqrt(c'Ωc) to ~0.3% at rank 30 for OCS-like c.
#     sqrt(c'Ωc) and Ωc cost O(n·RANK_OMEGA).
#   * The full-dense baseline uses the exact dense PEV.
# Comments further down that describe FΩ as an "exact MCMC factor" are
# historical.
# =============================================================================
# =============================================================================
# Norway Spruce -- Robust OCS (ROCS) via SQP with Dual Low-Rank Factorization
#
# Extends solve_full_dense()/solve_rsvd()/solve_from_factor() (as defined in
# spruce_single_gamma_comparison.jl and gamma_rank_sweep_spruce.jl) with the
# robust-OCS term from Eq. 29-30 of the manuscript's "Robust OCS" Methods
# addition, using the exact Ω factor built by build_robust_omega.jl
# (F_Ω = X/sqrt(S-1), from JWAS MCMC posterior draws of the same 3-trait
# index (Hjd17 + Htv17 - Sprant17)/3).
#
# REQUIRES: G, g_vec, all_ids already defined in this session -- run
# spruce_single_gamma_comparison.jl or gamma_rank_sweep_spruce.jl first.
#
# CRITICAL, NEW vs. the earlier scripts: F_Ω's individual ordering comes
# from MCMC_samples_EBV_*.txt's own column headers, which is a completely
# separate read path from geno_file's ID column (which all_ids/G/g_vec are
# aligned to). These two orderings are NOT assumed to match here -- they
# are explicitly realigned by ID below, with an added correlation check
# between g_vec and the Ω factor's own posterior-mean ĝ as a sanity check
# that both pipelines are really describing the same individuals/index.
# Do not skip Section 1.
#
# GAMMA/KAPPA CONVENTION: matches the existing scripts' parameterisation,
# where "gamma" relates to the manuscript's lambda by lambda = 2*gamma
# (i.e. minimize 0.5 x'Gx - (1/(2*gamma)) g'x <=> maximize g'x - gamma*x'Gx
# = maximize g'x - (lambda/2)*x'Gx). The robust objective
#   maximize c'g - kappa*sqrt(c'Omega*c) - gamma*c'Gc
# becomes, in the same minimize-form:
#   minimize 0.5 x'Gx - (1/(2*gamma))*g'x + (kappa/(2*gamma))*z
#   s.t. z >= sqrt(x'Omega*x)   (linearised via SQP tangent planes below)
#
# SCOPE NOTE: the fuller factorial breakdown (G alone vs. Ω alone vs. both
# decomposed) was cut from the manuscript for concision, so this script
# does not build all four combinations. It DOES build the single "no
# decomposition at all" baseline (dense G, materialised dense Omega) as
# an opt-in Section 6, since that's the actual headline speedup number
# Table 7 wants. It's expensive -- see Section 6's docstring/comments.
# =============================================================================

using LinearAlgebra, Random, Statistics, Printf, OSQP, JuMP, JLD2, DataFrames, CSV

# -----------------------------------------------------------------------
# 0. Guard: base script must already be loaded
# -----------------------------------------------------------------------

@isdefined(G)       || error("G is not defined -- run spruce_single_gamma_comparison.jl (or gamma_rank_sweep_spruce.jl) first.")
@isdefined(g_vec)   || error("g_vec is not defined -- run the base script first.")
@isdefined(all_ids) || error("all_ids is not defined -- run the base script first.")

RANK_G     = 30      # rank of the G (=H) factor
RANK_OMEGA = 30      # rank of the Ω (=PEV) factor (plus exact diagonal)
GAMMA      = 400.0   # HBLUP operating point (~73 of 5525 selected)

# -----------------------------------------------------------------------
# 1. Load Omega factor and realign to G/g_vec's individual order
# -----------------------------------------------------------------------

println("[1] Loading AI-REML PEV (Ω) aligned to the candidates in G...")
if !@isdefined(PEV) || size(PEV, 1) != length(all_ids)
    PEV = let f = jldopen(joinpath(@__DIR__, "..", "output", "AIREML_Hjd_17.jld2"), "r")
        ids_p = parse.(Int, f["ids"]); P = f["PEV"]; close(f)
        pos = Dict(id => k for (k, id) in enumerate(ids_p))
        ord = [pos[id] for id in all_ids]
        P[ord, ord]
    end
    GC.gc()
end
@printf("  PEV: %d x %d, mean diag=%.1f (SEP≈%.1f cm)\n\n", size(PEV)..., mean(diag(PEV)), sqrt(mean(diag(PEV))))

# -----------------------------------------------------------------------
# 2. Low-rank factor for G (fresh RSVD draw; independent of any factor
#    already computed by whichever base script was run)
# -----------------------------------------------------------------------

function rsvd(A::Matrix{T}, k::Int; p::Int=10, q::Int=2) where T
    n_ = size(A, 1)
    l = min(k + p, n_)
    R = randn(T, n_, l)
    Y = A * R
    for _ in 1:q
        Y = A * (A * Y)          # power iteration; A'=A since G is symmetric
    end
    Qm = Matrix(qr(Y).Q)[:, 1:l]
    B  = Symmetric(Qm' * A * Qm)   # small l x l SYMMETRIC matrix (two-sided
                                    # projection -- the one-sided Qm'*A used
                                    # in an earlier version of this function
                                    # only guarantees the right subspace, not
                                    # the correct eigenvectors within it, and
                                    # silently gives a wrong low-rank G)
    eig   = eigen(B)                # ascending eigenvalues by LAPACK convention
    order = sortperm(eig.values, rev = true)
    vals  = eig.values[order][1:k]
    vecs  = eig.vectors[:, order][:, 1:k]
    return Qm * vecs, vals
end

println("[2] Building G's low-rank factor (rank $RANK_G)...")
U_G, S_G = rsvd(G, RANK_G; p=10, q=2)
FG      = U_G * Diagonal(sqrt.(S_G))
DG_vals = diag(G) - vec(sum(FG.^2, dims=2))   # exact-diagonal correction, as in solve_rsvd

println("[2a] Building Ω's low-rank + diagonal factor (rank $RANK_OMEGA)...")
t_Ω = @elapsed begin
    U_Ω, S_Ω = rsvd(PEV, RANK_OMEGA; p=10, q=2)
    F_Ω = U_Ω * Diagonal(sqrt.(max.(S_Ω, 0.0)))
    d_Ω = max.(diag(PEV) .- vec(sum(F_Ω.^2, dims=2)), 0.0)
end
FΩ_aligned = (F = F_Ω, d = d_Ω)     # name kept so Sections 4-6 run unchanged
@printf("  done in %.2fs; min diagonal correction=%.1f\n\n", t_Ω, minimum(d_Ω))

# -----------------------------------------------------------------------
# 3. Solvers
# -----------------------------------------------------------------------

# Ω ≈ F F' + diag(d)  (see header)
omega_mul(Ωf, c) = Ωf.F * (Ωf.F' * c) .+ Ωf.d .* c
sd_omega(c, Ωf)  = sqrt(max(dot(c, omega_mul(Ωf, c)), 0.0))
sd_exact(c)      = sqrt(max(dot(c, PEV * c), 0.0))        # exact, dense -- reporting only

"""
    check_and_clean(c, label; warn_tol=1e-4, hard_tol=0.01)

Validate a returned contribution vector against sum(c)=1, c>=0 --
independent of, and not trusting, whatever the solver's own termination
status reports -- and return a corrected vector.

Two tiers, deliberately not one:
  - Violations larger than hard_tol are treated as a genuine correctness
    failure (seen empirically: sum(c)~9-12 under a loosened solver
    tolerance, i.e. orders of magnitude beyond noise) and are NOT
    auto-corrected -- this errors out rather than silently renormalizing
    a grossly wrong point into something that merely *looks* feasible.
  - Violations within hard_tol but above warn_tol (seen empirically:
    sum(c)=1.0 exactly but min(c) as low as -2e-4 when OSQP hits
    max_iter just shy of fully polishing near-zero contributions to
    exactly zero) are a standard, well-understood artifact of
    first-order QP solvers -- these are cleaned up by clipping negative
    entries to zero and renormalizing to sum exactly 1, which perturbs
    the solution by a negligible, sub-tolerance amount, and warned about
    so the correction is visible rather than silent.
"""
function check_and_clean(c::Vector{Float64}, label::AbstractString;
                          warn_tol::Float64=1e-4, hard_tol::Float64=0.01)
    s = sum(c)
    minc = minimum(c)
    if abs(s - 1.0) > hard_tol || minc < -hard_tol
        error("$label: SEVERELY INFEASIBLE -- sum(c)=$(round(s, digits=4)), min(c)=$(round(minc, digits=6)) (hard_tol=$hard_tol). Refusing to auto-correct -- this is not a minor numerical residual. Investigate before proceeding.")
    end
    if abs(s - 1.0) > warn_tol || minc < -warn_tol
        @warn "$label: minor infeasibility (sum(c)=$(round(s, digits=6)), min(c)=$(round(minc, digits=8))) -- auto-corrected via clip+renormalize."
    end
    c_clean = max.(c, 0.0)
    c_clean ./= sum(c_clean)
    return c_clean
end

"""Standard (kappa=0) low-rank OCS, same formulation as solve_from_factor
in the base scripts."""
function solve_ocs_lowrank(g, FG, DG_vals; gamma, osqp_eps::Float64=1e-6, osqp_max_iter::Int=50000)
    n_, rankG = size(FG)
    model = Model(optimizer_with_attributes(OSQP.Optimizer,
        "max_iter" => osqp_max_iter, "eps_abs" => osqp_eps, "eps_rel" => osqp_eps,
        "polish" => true, "verbose" => false))
    @variable(model, x[1:n_] >= 0)
    @variable(model, y[1:rankG])
    @objective(model, Min,
        0.5 * sum(DG_vals[i] * x[i]^2 for i in 1:n_) + 0.5 * dot(y, y) - (1/(2*gamma)) * dot(g, x))
    @constraint(model, sum(x) == 1.0)
    for j in 1:rankG
        @constraint(model, y[j] == sum(FG[i, j] * x[i] for i in 1:n_))
    end
    t = @elapsed JuMP.optimize!(model)
    c = check_and_clean(JuMP.value.(x), "solve_ocs_lowrank(gamma=$gamma)")
    return (contributions = c, status = string(termination_status(model)), time_opt = t)
end

# Correctness check: at rank 30 this should closely match the established
# full-dense gamma=300 result (gain≈65.09, coancestry≈0.0376, sel≈78 from
# gamma_rank_sweep_spruce.jl). If it doesn't, stop here -- something's still
# wrong upstream (G's factor, g_vec, or the alignment) -- rather than
# proceeding to the kappa scan with an untrustworthy G factor.
println("[2b] Correctness check: rank-$RANK_G OCS (kappa=0 equivalent) vs. established full-dense result...")
_check = solve_ocs_lowrank(g_vec, FG, DG_vals; gamma = GAMMA)
_check_gain = dot(_check.contributions, g_vec)
_check_coan = _check.contributions' * G * _check.contributions
_check_nsel = sum(_check.contributions .> 1e-4)
@printf("  rank-%d OCS at gamma=%.0f: gain=%.4f  coancestry=%.6f  n_sel=%d\n",
        RANK_G, GAMMA, _check_gain, _check_coan, _check_nsel)
if @isdefined(baseline) && hasproperty(baseline, :genetic_gain)
    @printf("  full-dense baseline from base script: gain=%.4f  coancestry=%.6f  n_sel=%d\n\n",
            baseline.genetic_gain, baseline.genetic_var, sum(baseline.contributions .> 1e-4))
else
    println("  (run spruce_single_gamma_comparison_hblup.jl first to compare with the full-dense baseline)\n")
end

"""
    solve_robust_rsvd(g, FG, DG_vals, FΩ; gamma, kappa, K=15, tol=1e-6, verbose=true)

Robust OCS (ROCS) via SQP with dual low-rank factorization (Eq. 29-32,
Robust OCS Methods addition). Tangent-plane cuts are ACCUMULATED across
SQP iterations rather than replaced: sqrt(c'Ωc) is convex (it's the norm
of F_Ω'c), so every tangent plane is a *global* underestimate, not just a
local one -- this is the standard cutting-plane treatment of a convex
constraint, and accumulating cuts (vs. keeping only the latest) is what
gives monotonic convergence rather than possible oscillation.

kappa=0 is special-cased to call solve_ocs_lowrank directly rather than
going through the SQP/z machinery at all: z has zero weight in the
objective at kappa=0, making it a genuinely free variable, which is a
textbook hard case for ADMM (no gradient pressure on z at all, so OSQP
can spend many iterations without converging to tight tolerance even
though x's optimum is unaffected). Bypassing this is both exact (we
already proved kappa=0 is mathematically identical to standard OCS) and
avoids the pathological case entirely rather than papering over it.

Returns: contributions, status, n_iter, converged, sd_uncertainty (exact,
via sd_omega), gain, coancestry (both computed against the TRUE g/G, not
the low-rank surrogate, matching genetic_gain/genetic_var reporting
convention in the base scripts), time_opt.
"""
function solve_robust_rsvd(g, G, FG, DG_vals, FΩ; gamma, kappa, K::Int=15, tol::Float64=1e-6,
                            osqp_eps::Float64=1e-6, osqp_max_iter::Int=50000, verbose::Bool=true)
    n_, rankG = size(FG)

    if kappa == 0.0
        sol = solve_ocs_lowrank(g, FG, DG_vals; gamma = gamma, osqp_eps = osqp_eps, osqp_max_iter = osqp_max_iter)
        return (
            contributions  = sol.contributions,
            status         = sol.status,
            n_iter         = 0,
            converged      = true,
            sd_uncertainty = sd_omega(sol.contributions, FΩ),
            gain           = dot(sol.contributions, g),
            coancestry     = sol.contributions' * G * sol.contributions,
            time_opt       = sol.time_opt
        )
    end

    t_opt = @elapsed begin
        c_prev = solve_ocs_lowrank(g, FG, DG_vals; gamma = gamma, osqp_eps = osqp_eps, osqp_max_iter = osqp_max_iter).contributions

        cuts = Tuple{Vector{Float64},Float64}[]   # (v_k, s_k) accumulated so far
        n_iter = 0
        converged = false
        local model, status_str

        for k in 0:(K-1)
            v_k = omega_mul(FΩ, c_prev)                  # Ω c_(k), O(n·RANK_OMEGA)
            s_k = sd_omega(c_prev, FΩ)                    # = sqrt(c_(k)'Ωc_(k)) >= 0, exact
            s_k < 1e-6 && @warn "solve_robust_rsvd(gamma=$gamma, kappa=$kappa, iter=$(k+1)): s_k=$s_k is very small (near-zero projected uncertainty at this iterate)."
            push!(cuts, (v_k, s_k))

            # Rebuilt FRESH (cold-start) each iteration with ALL cuts so far,
            # rather than reusing one model warm-started across constraint
            # additions: empirically, the latter was NOT reliably reaching
            # the true optimum here -- kappa=0 (where z has zero objective
            # weight and can't affect x at all) still gave a wrong answer,
            # and every kappa hit the max iteration count without the
            # convergence check ever firing, both signs OSQP was returning
            # a stale/non-optimal point rather than genuinely failing to
            # converge mathematically.
            model = Model(optimizer_with_attributes(OSQP.Optimizer,
                "max_iter" => osqp_max_iter, "eps_abs" => osqp_eps, "eps_rel" => osqp_eps,
                "polish" => true, "verbose" => false))
            @variable(model, x[1:n_] >= 0)
            @variable(model, y[1:rankG])
            @variable(model, z >= 0)
            @objective(model, Min,
                0.5 * sum(DG_vals[i] * x[i]^2 for i in 1:n_) + 0.5 * dot(y, y)
                - (1/(2*gamma)) * dot(g, x) + (kappa/(2*gamma)) * z)
            @constraint(model, sum(x) == 1.0)
            for j in 1:rankG
                @constraint(model, y[j] == sum(FG[i, j] * x[i] for i in 1:n_))
            end
            for (vc, sc) in cuts
                # Written as sc*z >= dot(vc,x), NOT z >= dot(vc,x)/sc: the
                # latter divides every one of vc's ~n entries by sc, which
                # for small sc inflates the whole row into a wildly
                # different scale from the rest of the problem -- this was
                # the cause of a "qdldl: error forming and permuting KKT
                # matrix" crash. Multiplying z's own single coefficient by
                # sc instead is mathematically identical (sc>=0) but never
                # blows up more than one entry.
                @constraint(model, sc * z >= dot(vc, x))
            end

            JuMP.optimize!(model)
            status_str = string(termination_status(model))
            status_str in ("OPTIMAL", "ALMOST_OPTIMAL") ||
                @warn "SQP iter $(k+1) (kappa=$kappa): OSQP status=$status_str, not OPTIMAL -- result may not be trustworthy."

            c_new = check_and_clean(JuMP.value.(x), "solve_robust_rsvd(gamma=$gamma, kappa=$kappa, iter=$(k+1))")
            n_iter = k + 1

            delta = maximum(abs.(c_new .- c_prev))
            verbose && @printf("    SQP iter %2d: Δc_max=%.2e  z=%.4f  true SD=%.4f  status=%s\n",
                                n_iter, delta, JuMP.value(z), sd_omega(c_new, FΩ), status_str)
            c_prev = c_new
            if delta < tol
                converged = true
                break
            end
        end
    end

    c_final = c_prev
    return (
        contributions  = c_final,
        status         = status_str,
        n_iter         = n_iter,
        converged      = converged,
        sd_uncertainty = sd_omega(c_final, FΩ),
        gain           = dot(c_final, g),
        coancestry     = c_final' * G * c_final,
        time_opt       = t_opt
    )
end

"""
    solve_full_dense_robust(g, G, FΩ; gamma, kappa, K, tol=1e-6, verbose=true)

The "no decomposition at all" baseline for the ROCS speedup number: same
SQP cutting-plane scheme as solve_robust_rsvd, but with G used directly
(dense x'Gx, no factor) and Ω materialised explicitly as
Ω_dense = FΩ*FΩ' before the loop, so every per-iterate tangent-plane
evaluation costs O(n^2) instead of O(nS). The Ω_dense materialisation
(O(n^2 S) flops, ~$(round(Int, 5525^2*8/1e6)) MB for n=5525) is included
in the reported time, since avoiding exactly this cost is the point of
the factorized route.

Because FΩ is EXACT (not an RSVD approximation) for this dataset, the
iterate sequence here should be mathematically identical to
solve_robust_rsvd's at the same (gamma, kappa) -- so this run doubles as
a correctness check on the factorized implementation, not just a timing
baseline. If sd_uncertainty or contributions differ from
solve_robust_rsvd's by more than solver tolerance, treat that as a bug
to investigate, not approximation error.

EXPENSIVE: each SQP iteration re-solves a dense n x n QP. Pass
K = (the already-converged n_iter from a prior solve_robust_rsvd call)
+ a small buffer, rather than guessing -- the iterate sequence should
converge in about the same number of steps either way. Consider running
verbose=true and watching the per-iteration Δc_max before committing to
multiple replicates.
"""
function solve_full_dense_robust(g, G, Ω_dense; gamma, kappa, K::Int, tol::Float64=1e-6, verbose::Bool=true)
    n_ = length(g)
    t_mat = 0.0   # Ω (=PEV) is already dense; nothing to materialise

    t_solve = @elapsed begin
        # First solve (no z-cuts yet) seeds c_(0); equivalent to kappa=0 OCS
        # since z is unconstrained by x at this point.
        model0 = Model(optimizer_with_attributes(OSQP.Optimizer,
            "max_iter" => 50000, "eps_abs" => 1e-6, "eps_rel" => 1e-6, "polish" => true, "verbose" => false))
        @variable(model0, x0[1:n_] >= 0)
        @objective(model0, Min, 0.5 * (x0' * G * x0) - (1/(2*gamma)) * dot(g, x0))
        @constraint(model0, sum(x0) == 1.0)
        JuMP.optimize!(model0)
        c_prev = check_and_clean(JuMP.value.(x0), "solve_full_dense_robust(gamma=$gamma) seed")

        cuts = Tuple{Vector{Float64},Float64}[]
        n_iter = 0
        converged = false
        local model, status_str

        for k in 0:(K-1)
            v_k = Ω_dense * c_prev                          # O(n^2) -- the cost the factor avoids
            s_k = sqrt(max(dot(c_prev, v_k), 0.0))          # >= 0, exact
            s_k < 1e-6 && @warn "solve_full_dense_robust(gamma=$gamma, kappa=$kappa, iter=$(k+1)): s_k=$s_k is very small."
            push!(cuts, (v_k, s_k))

            # Rebuilt fresh each iteration -- see solve_robust_rsvd for why
            # (reusing one model warm-started across constraint additions
            # was empirically not reaching the true optimum).
            model = Model(optimizer_with_attributes(OSQP.Optimizer,
                "max_iter" => 50000, "eps_abs" => 1e-6, "eps_rel" => 1e-6, "polish" => true, "verbose" => false))
            @variable(model, x[1:n_] >= 0)
            @variable(model, z >= 0)
            @objective(model, Min, 0.5 * (x' * G * x) - (1/(2*gamma)) * dot(g, x) + (kappa/(2*gamma)) * z)
            @constraint(model, sum(x) == 1.0)
            for (vc, sc) in cuts
                @constraint(model, sc * z >= dot(vc, x))   # see solve_robust_rsvd for why not z >= dot(vc,x)/sc
            end

            JuMP.optimize!(model)
            status_str = string(termination_status(model))
            status_str in ("OPTIMAL", "ALMOST_OPTIMAL") ||
                @warn "[dense] SQP iter $(k+1) (kappa=$kappa): OSQP status=$status_str, not OPTIMAL."

            c_new = check_and_clean(JuMP.value.(x), "solve_full_dense_robust(gamma=$gamma, kappa=$kappa, iter=$(k+1))")
            n_iter = k + 1

            delta = maximum(abs.(c_new .- c_prev))
            verbose && @printf("    [dense] SQP iter %2d: Δc_max=%.2e  status=%s\n", n_iter, delta, status_str)
            c_prev = c_new
            if delta < tol
                converged = true
                break
            end
        end
    end

    c_final = c_prev
    return (
        contributions  = c_final,
        status         = status_str,
        n_iter         = n_iter,
        converged      = converged,
        sd_uncertainty = sqrt(dot(c_final, Ω_dense * c_final)),
        gain           = dot(c_final, g),
        coancestry     = c_final' * G * c_final,
        time_mat       = t_mat,
        time_solve     = t_solve,
        time_total     = t_mat + t_solve
    )
end

# -----------------------------------------------------------------------
# 4. Kappa pilot scan -- run this first (mirrors the RUN_PILOT gamma scan
#    pattern in gamma_rank_sweep_spruce.jl). No natural scale for kappa is
#    known in advance, so this brackets a wide geometric range; inspect
#    how SD/gain/coancestry move and narrow from there.
# -----------------------------------------------------------------------

RUN_KAPPA_PILOT = false
K_PILOT = 6   # deliberately small: this scan is for locating roughly where
              # kappa* sits, not for a converged Table 7 number. Each
              # kappa here rebuilds up to K_PILOT+1 full JuMP/OSQP models
              # (see solve_robust_rsvd), which is the main memory/time cost
              # of this loop -- keep this low, and use the default K=15 in
              # Section 5 for the final committed (gamma*, kappa*) pair.

if RUN_KAPPA_PILOT
    println("="^70)
    println("PILOT: Gain / SD / Coancestry across a wide kappa range (gamma=$GAMMA)")
    println("="^70)
    kappa_pilot = [0.0, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
    println(rpad("kappa", 10), rpad("gain", 12), rpad("SD", 12),
            rpad("coancestry", 14), rpad("n_iter", 8), rpad("n_sel", 8), "status")
    println("-"^80)
    for kap in kappa_pilot
        sol = solve_robust_rsvd(g_vec, G, FG, DG_vals, FΩ_aligned;
                                 gamma = GAMMA, kappa = kap, K = K_PILOT, verbose = false)
        n_sel = sum(sol.contributions .> 1e-4)
        @printf("%-10.3g%-12.4f%-12.4f%-14.6f%-8d%-8d%s\n",
                kap, sol.gain, sol.sd_uncertainty, sol.coancestry, sol.n_iter, n_sel, sol.status)
        GC.gc()   # release each kappa's ~K_PILOT JuMP/OSQP models before the next
    end
    println()
    println("Pick kappa* from where SD/coancestry visibly trade off against gain,")
    println("then set RUN_KAPPA_PILOT = false and run Section 5 for the Table 7 pair.")
    println()
end


# -----------------------------------------------------------------------
# 5. OCS vs ROCS comparison at a chosen kappa (Table 7 pair)
# -----------------------------------------------------------------------

if !RUN_KAPPA_PILOT
    KAPPA_STAR = 1.0   # <-- set from the pilot scan above before running this section

    println("="^70)
    println("OCS vs ROCS at gamma=$GAMMA, kappa=$KAPPA_STAR")
    println("="^70)

    ocs  = solve_ocs_lowrank(g_vec, FG, DG_vals; gamma = GAMMA)
    rocs = solve_robust_rsvd(g_vec, G, FG, DG_vals, FΩ_aligned;
                              gamma = GAMMA, kappa = KAPPA_STAR, verbose = true)

    results_df = DataFrame(
        method     = ["OCS", "ROCS"],
        gamma      = [GAMMA, GAMMA],
        kappa      = [0.0, KAPPA_STAR],
        gain       = [dot(ocs.contributions, g_vec), rocs.gain],
        sd         = [sd_omega(ocs.contributions, FΩ_aligned), rocs.sd_uncertainty],
        sd_exact   = [sd_exact(ocs.contributions), sd_exact(rocs.contributions)],
        coancestry = [ocs.contributions' * G * ocs.contributions, rocs.coancestry],
        n_selected = [sum(ocs.contributions .> 1e-4), sum(rocs.contributions .> 1e-4)]
    )
    println(results_df)

    OUT_DIR_ROBUST = normpath(joinpath(@__DIR__, "..", "output", "robust_ocs"))
    mkpath(OUT_DIR_ROBUST)
    out_csv = joinpath(OUT_DIR_ROBUST, "ocs_vs_rocs_gamma$(Int(GAMMA))_kappa$(KAPPA_STAR).csv")
    CSV.write(out_csv, results_df)
    @printf("\nSaved: %s\n", out_csv)
end

# -----------------------------------------------------------------------
# 6. Speedup vs. full-dense ROCS baseline ("no decomposition at all") --
#    the headline number Table 7's speedup column actually wants.
#
#    EXPENSIVE: each SQP iteration re-solves a dense n x n (n=5525) QP,
#    so expect this to take substantially longer than Section 5 -- your
#    existing full-dense OCS baseline alone has run 100-200+s per solve
#    elsewhere in this dataset, and this repeats a solve of that order
#    K_dense times. Opt-in via the flag below. Run Section 5 first so
#    `rocs` (used to set K, and as the comparison point) is in scope.
#
#    F_Ω is the EXACT factor (not an RSVD approximation) for this dataset,
#    but F_G is NOT (it's the same rank-RANK_G RSVD approximation used
#    throughout this manuscript) -- so the two solves optimize slightly
#    different objectives by design (true dense G vs. its rank-RANK_G
#    approximation), and solver-tolerance agreement is neither expected
#    nor the right bar. Any discrepancy should be of the same order as
#    G's already-characterized approximation error (~0.1-1% gain_diff at
#    rank 30 elsewhere in this manuscript); the check below flags only
#    disagreement well beyond that established range.
# -----------------------------------------------------------------------

RUN_DENSE_ROCS_BASELINE = true

if !RUN_KAPPA_PILOT && RUN_DENSE_ROCS_BASELINE
    @isdefined(rocs) || error("Run Section 5 first (with RUN_KAPPA_PILOT=false) -- `rocs` is not defined.")

    println("="^70)
    println("Full-dense ROCS baseline (no G or Ω decomposition) -- this may take a while")
    println("="^70)
    K_dense = rocs.n_iter + 2   # dense iterate sequence should match rocs' almost exactly
    @printf("  Using K=%d (factorized run converged in %d iterations at the same gamma/kappa)\n\n",
            K_dense, rocs.n_iter)

    dense = solve_full_dense_robust(g_vec, G, PEV;
                                     gamma = GAMMA, kappa = KAPPA_STAR, K = K_dense, verbose = true)

    speedup = dense.time_total / rocs.time_opt
    @printf("\n  Dense total time  : %.2fs  (materialise Ω: %.2fs + solve: %.2fs)\n",
            dense.time_total, dense.time_mat, dense.time_solve)
    @printf("  Factorized time   : %.2fs  (+ one-off RSVD of G: see [2]; of Ω: %.2fs)\n", rocs.time_opt, t_Ω)
    @printf("  Speedup           : %.1fx\n\n", speedup)

    sd_diff       = abs(dense.sd_uncertainty - sd_exact(rocs.contributions))
    c_diff        = maximum(abs.(dense.contributions .- rocs.contributions))
    gain_diff_pct = abs(dense.gain - rocs.gain) / abs(dense.gain) * 100
    @printf("  Dense gain: %.4f   Factorized gain: %.4f\n", dense.gain, rocs.gain)
    @printf("  gain_diff_pct (matching Table 2-4 convention) = %.3f%%\n", gain_diff_pct)
    @printf("  |SD_dense - SD_factorized|  = %.2e\n", sd_diff)
    @printf("  max|c_dense - c_factorized| = %.2e\n", c_diff)
    # NOTE: F_Ω is exact here, but F_G (rank RANK_G) is NOT -- so exact
    # (solver-tolerance) agreement is neither expected nor the right bar.
    # The two solves optimize slightly different objectives by design (one
    # uses the true dense G, the other its rank-RANK_G approximation), so
    # any disagreement should be of the same order as the already-
    # characterized G-only approximation error (~0.1-1% gain_diff at rank
    # 30 elsewhere in this manuscript), not solver-tolerance-level (1e-6).
    # Flag only if it's well beyond that established range.
    if gain_diff_pct > 2.0
        @warn "gain_diff_pct=$(round(gain_diff_pct, digits=3))% is well beyond the ~0.1-1% range already established for rank-$RANK_G G alone elsewhere in this manuscript -- this is larger than G's approximation error should account for. Investigate solve_robust_rsvd/solve_full_dense_robust before reporting the speedup number."
    else
        println("  gain_diff_pct is consistent with rank-$RANK_G G's already-characterized approximation error -- no new discrepancy from the robust (Ω) extension. Speedup number is trustworthy.")
    end

    results_df.speedup       = [missing, speedup]
    results_df.gain_diff_pct = [missing, gain_diff_pct]
    CSV.write(out_csv, results_df)
    @printf("\nUpdated: %s\n", out_csv)
end
