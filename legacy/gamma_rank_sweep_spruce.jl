# =============================================================================
# Norway Spruce — Gamma x Rank Sweep for Theoretical Speedup Analysis
#
# Adapted from the QTL-MAS 2010 gamma x rank sweep script. Changes from that
# version:
#   1. No sex constraint. Norway spruce trees are monoecious (each tree can
#      act as both seed and pollen parent), so the QTL-MAS-style separate
#      male/female equality constraints do not apply. Replaced with a single
#      sum(c) = 1 budget constraint.
#   2. Breeding value input is a 3-trait index, (Hjd17 + Htv17 - Sprant17)/3,
#      built from three separate per-trait EBV files, matching the index
#      used in Ahlinder & Waldmann (2026) [\citep{Ahlinder2026}].
#   3. Gamma grid: WIDENED relative to the QTL-MAS script pending a pilot run
#      (see Section 1b below) -- the QTL-MAS grid was tuned empirically for
#      that dataset's G/GEBV scale, which does not necessarily transfer here,
#      especially given the sum(c)=1 rescaling (see chat discussion). Run the
#      pilot first, inspect n_selected per gamma, then narrow this list.
#
# Research question:
#   Does tightening the diversity constraint (lower gamma) increase the
#   computational advantage of RSVD-OCS by concentrating the OCS solution
#   on fewer individuals, making the (H-)relationship matrix better
#   approximated at low rank -- as already shown for QTL-MAS -- on a real,
#   larger (n=5,525) breeding population too?
#
# Outputs (same schema as the QTL-MAS script, for direct reuse of the R
# plotting code):
#   gamma_rank_sweep_raw.csv     -- one row per gamma x rank x replicate
#   gamma_rank_sweep_summary.csv -- mean/sd across replicates, one row per gamma x rank
#   gamma_rank_sweep_elbow.csv   -- recommended rank and speedup at elbow, one row per gamma
#
# OPEN QUESTIONS -- confirm before trusting the output (see chat):
#   - EBV_Hjd17.txt / EBV_Htv17.txt / EBV_Sprant17.txt confirmed as
#     comma-delimited with header ID,EBV,PEV. Loading code below matches this.
#   - geno_file confirmed to have an ID column (col 1) followed by the n x n
#     relationship matrix -- row i's self-relationship (1.0) sits at column
#     i+1, confirming row order = ID column order. IDs are read directly from
#     this file and used to align g_vec, so no ordering assumption is needed
#     between geno_file and the EBV files.
#   - geno_file uses tau=1, omega=1 (confirmed intentional -- the original
#     Legarra et al. 2009 H-matrix formulation). Manuscript Methods text
#     currently says tau=1, omega=0 and needs updating to match.
# =============================================================================

using LinearAlgebra
using Random
using Statistics
using Printf
using OSQP
using JuMP
using DelimitedFiles
using DataFrames
using CSV
using Dates

# =============================================================================
# 0. CONFIGURATION
# =============================================================================

BASE_DIR = "EDIT_ME/NorwaySpruceData/"
OUT_DIR  = "EDIT_ME/NorwaySpruceData/results_gamma_rank_sweep/"

gebv_file_hjd17    = joinpath(BASE_DIR, "Save/MCMC_H55253/EBV_Hjd17.txt")
gebv_file_htv17    = joinpath(BASE_DIR, "Save/MCMC_H55253/EBV_Htv17.txt")
gebv_file_sprant17 = joinpath(BASE_DIR, "Save/MCMC_H55253/EBV_Sprant17.txt")

# tau=1, omega=1: confirmed intentional -- this is the original Legarra et al.
# (2009) H-matrix formulation (also AGHmatrix's "Martini" method default), not
# a nonstandard choice. NOTE: manuscript Methods currently states tau=1,
# omega=0 -- that text needs to be updated to tau=1, omega=1 to match what
# this analysis (and, presumably, the rest of the spruce case study) actually uses.
geno_file = joinpath(BASE_DIR, "Save/Hmat_5525_spruce_tau_1_omega_1_PDF.txt")

# Selection-index weights: (Hjd17 + Htv17 - Sprant17) / 3, matching
# Ahlinder & Waldmann (2026). Equal weights, sign-flip on the defect trait,
# averaged (not just summed) over the three traits.
INDEX_WEIGHTS = (hjd17 = 1/3, htv17 = 1/3, sprant17 = -1/3)

SEL_THRESH  = 1e-4
GAIN_THRESH = 1.0      # % gain difference threshold for recommended rank

# --- Gamma grid: narrowed from the pilot run (2026-08-04). Pilot showed the
# degenerate-to-diffuse transition spans gamma~100 (34 selected) to gamma~1000
# (205 selected); this grid brackets that zone, centered on the requested 300.
gammas = [100.0, 200.0, 300.0, 500.0, 1000.0]

# Ranks to test at each gamma
ranks = [5, 10, 15, 20, 25, 30, 40, 50, 75, 100]

# RSVD replicates per gamma x rank
n_replicates = 5

# =============================================================================
# 1. DATA LOADING
# =============================================================================

println("="^70)
println("Norway Spruce -- Gamma x Rank Sweep")
println("="^70)
println("Timestamp: $(Dates.now())")
println()

println("[1] Loading EBV files and building selection index...")

# Files are comma-delimited with header: ID,EBV,PEV
hjd_df    = CSV.read(gebv_file_hjd17,    DataFrame; delim=',')
htv_df    = CSV.read(gebv_file_htv17,    DataFrame; delim=',')
sprant_df = CSV.read(gebv_file_sprant17, DataFrame; delim=',')

@assert names(hjd_df) == names(htv_df) == names(sprant_df) "EBV files have different column names -- check format assumption"

# Join on ID (inner join -- individuals must have all three trait EBVs to
# get an index value; report how many were dropped, if any). Only ID and
# EBV are needed here; PEV is dropped.
idx_df = innerjoin(
    rename(hjd_df[:, [:ID, :EBV]],    :EBV => :EBV_hjd17),
    rename(htv_df[:, [:ID, :EBV]],    :EBV => :EBV_htv17),
    on = :ID
)
idx_df = innerjoin(
    idx_df,
    rename(sprant_df[:, [:ID, :EBV]], :EBV => :EBV_sprant17),
    on = :ID
)

n_dropped = nrow(hjd_df) - nrow(idx_df)
if n_dropped != 0
    @warn "Index join dropped $n_dropped individuals not present in all three EBV files -- check ID overlap."
end

idx_df.index = INDEX_WEIGHTS.hjd17    .* idx_df.EBV_hjd17 .+
               INDEX_WEIGHTS.htv17    .* idx_df.EBV_htv17 .+
               INDEX_WEIGHTS.sprant17 .* idx_df.EBV_sprant17

id_to_g = Dict(Int(idx_df.ID[i]) => idx_df.index[i] for i in eachindex(idx_df.ID))

@printf("  n=%d individuals with a complete 3-trait index (pre-alignment)\n", nrow(idx_df))
@printf("  index range: [%.3f, %.3f]  mean=%.3f  sd=%.3f\n\n",
        minimum(idx_df.index), maximum(idx_df.index), mean(idx_df.index), std(idx_df.index))

println("[2] Building relationship matrix and aligning IDs...")
M = readdlm(geno_file, ',', header=false)

# geno_file has an ID column (col 1) followed by the n x n relationship
# matrix. Verified: row i's self-relationship (1.0) sits at column i+1,
# confirming row order == ID column order. Use these IDs as the authoritative
# ordering -- do NOT assume it matches the EBV file order.
all_ids = Int.(round.(M[:, 1]))
n_total = length(all_ids)
G       = Matrix(M[:, 2:end])
@assert size(G, 1) == size(G, 2) == n_total "geno_file is not square after dropping the ID column " *
    "($(size(G,1)) x $(size(G,2)) vs n=$n_total IDs) -- check file format."

# Spot-check: diagonal should be ~1.0 at position (i, i) if row order matches
# the ID column as expected.
diag_check = diag(G)
@assert all(abs.(diag_check .- 1.0) .< 0.3) "geno_file diagonal is not close to 1.0 -- " *
    "row/column order may not match the ID column. Aborting rather than silently misaligning."

G = (G + G') / 2   # enforce symmetry against any I/O rounding asymmetry

missing_ids = [id for id in all_ids if !haskey(id_to_g, id)]
if !isempty(missing_ids)
    @warn "$(length(missing_ids)) individuals in geno_file have no 3-trait index (missing from one or more EBV files) -- excluding them from the analysis."
end

keep_mask = [haskey(id_to_g, id) for id in all_ids]
all_ids   = all_ids[keep_mask]
G         = G[keep_mask, keep_mask]
n_total   = length(all_ids)
g_vec     = [id_to_g[id] for id in all_ids]   # order now == G's row/col order, by construction

@printf("  Relationship matrix (aligned): %d x %d  (diag mean=%.4f)\n", n_total, n_total, mean(diag(G)))
@printf("  index range (aligned): [%.3f, %.3f]  mean=%.3f  sd=%.3f\n\n",
        minimum(g_vec), maximum(g_vec), mean(g_vec), std(g_vec))

# =============================================================================
# 1b. PILOT RUN -- find a sensible gamma range before committing to the full
#     sweep with replicates. Run this block first, inspect n_selected, then
#     edit `gammas` above accordingly (aim to bracket the transition from a
#     near-degenerate few-individual solution to a diffuse, most-of-the-
#     population solution, as was done for QTL-MAS).
# =============================================================================

RUN_PILOT = false

if RUN_PILOT
    println("="^70)
    println("PILOT: full-dense solution size across a wide gamma range")
    println("="^70)
    pilot_gammas = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 1e2, 1e3, 1e4]
    for gam in pilot_gammas
        model = Model(optimizer_with_attributes(OSQP.Optimizer,
            "max_iter" => 20000, "eps_abs" => 1e-6, "eps_rel" => 1e-6, "verbose" => false))
        @variable(model, x[1:n_total] >= 0)
        @objective(model, Min, 0.5 * (x' * G * x) - (1/(2*gam)) * dot(g_vec, x))
        @constraint(model, sum(x) == 1.0)
        JuMP.optimize!(model)
        c = JuMP.value.(x)
        n_sel = sum(c .> SEL_THRESH)
        @printf("  gamma=%-10.4g  status=%-10s  n_selected=%d\n",
                gam, string(termination_status(model)), n_sel)
    end
    println()
    println("  Pick `gammas` above to bracket where n_selected moves from ~1-5")
    println("  (degenerate) through a moderate range to most of n=$n_total (diffuse),")
    println("  then set RUN_PILOT = false and re-run for the full sweep.")
    println()
end

# =============================================================================
# 2. SOLVER FUNCTIONS  (sum(c) = 1 budget constraint, no sex split)
# =============================================================================

function rsvd(A::Matrix{T}, k::Int; p::Int=10, q::Int=2) where T
    n, m = size(A)
    l    = min(k + p, min(n, m))
    Omega = randn(T, m, l)
    Y    = A * Omega
    for _ in 1:q
        Y = A * (A' * Y)
    end
    Q, _ = qr(Y)
    Q    = Matrix(Q)
    B    = Q' * A
    U_tilde, S, V = svd(B)
    U    = Q * U_tilde
    return U[:, 1:k], S[1:k], V[:, 1:k]
end

"""Full dense OCS. sum(c) = 1, no sex constraint. Returns named tuple."""
function solve_full_dense(G, g; gamma)
    n   = length(g)
    t   = @elapsed begin
        model = Model(optimizer_with_attributes(OSQP.Optimizer,
            "max_iter" => 20000, "eps_abs" => 1e-6,
            "eps_rel" => 1e-6, "verbose" => false))
        @variable(model, x[1:n] >= 0)
        @objective(model, Min,
            0.5 * (x' * G * x) - (1/(2*gamma)) * dot(g, x))
        @constraint(model, sum(x) == 1.0)
        JuMP.optimize!(model)
        c = JuMP.value.(x)
    end
    return (
        contributions = c,
        status        = string(termination_status(model)),
        genetic_gain  = dot(c, g),
        genetic_var   = c' * G * c,
        time_total    = t
    )
end

"""RSVD OCS with auxiliary variable formulation. sum(c) = 1, no sex constraint."""
function solve_rsvd(G, g; rank, gamma)
    n   = length(g)

    t_decomp = @elapsed begin
        U, S, _ = rsvd(G, rank; p=10, q=2)
        F       = U * Diagonal(sqrt.(S))          # n x rank
        D_vals  = diag(G) - vec(sum(F.^2, dims=2))
    end

    t_opt = @elapsed begin
        model = Model(optimizer_with_attributes(OSQP.Optimizer,
            "max_iter" => 20000, "eps_abs" => 1e-6,
            "eps_rel" => 1e-6, "verbose" => false))
        @variable(model, x[1:n] >= 0)
        @variable(model, y[1:rank])
        @objective(model, Min,
            0.5 * sum(D_vals[i] * x[i]^2 for i in 1:n) +
            0.5 * dot(y, y) -
            (1/(2*gamma)) * dot(g, x))
        @constraint(model, sum(x) == 1.0)
        for j in 1:rank
            @constraint(model, y[j] == sum(F[i, j] * x[i] for i in 1:n))
        end
        JuMP.optimize!(model)
        c = JuMP.value.(x)
    end

    return (
        contributions = c,
        status        = string(termination_status(model)),
        genetic_gain  = dot(c, g),
        genetic_var   = c' * G * c,
        time_decomp   = t_decomp,
        time_opt      = t_opt,
        time_total    = t_decomp + t_opt
    )
end

"""Compute comparison metrics between baseline and approximate solution."""
function compare(baseline, approx, G)
    gain_diff_pct = abs(approx.genetic_gain - baseline.genetic_gain) /
                    abs(baseline.genetic_gain) * 100
    var_diff_pct  = abs(approx.genetic_var - baseline.genetic_var) /
                    abs(baseline.genetic_var) * 100
    speedup       = baseline.time_total / approx.time_total

    rb = sortperm(sortperm(baseline.contributions))
    ra = sortperm(sortperm(approx.contributions))
    spearman_rho  = cor(Float64.(rb), Float64.(ra))

    sel_b   = findall(baseline.contributions .> SEL_THRESH)
    sel_a   = findall(approx.contributions   .> SEL_THRESH)
    overlap = length(intersect(sel_b, sel_a)) / max(length(sel_b), 1) * 100

    return (
        gain_diff_pct = gain_diff_pct,
        var_diff_pct  = var_diff_pct,
        speedup       = speedup,
        spearman_rho  = spearman_rho,
        overlap_pct   = overlap
    )
end

# =============================================================================
# 3. MAIN SWEEP
# =============================================================================

if !RUN_PILOT

println("="^70)
println("Running gamma x rank sweep")
println("="^70)
println()
println("Gammas : $gammas")
println("Ranks  : $ranks")
println("Reps   : $n_replicates")
println()

raw_rows = []

for gamma in gammas

    println("-"^70)
    @printf("  gamma = %.4g\n", gamma)
    println("-"^70)

    print("  Full Dense baseline ... ")
    baseline = solve_full_dense(G, g_vec; gamma=gamma)

    if baseline.status != "OPTIMAL"
        @printf("FAILED (%s) -- skipping gamma\n\n", baseline.status)
        continue
    end

    n_sel = sum(baseline.contributions .> SEL_THRESH)
    @printf("done (%.1fs)  gain=%.4f  var=%.6f  sel=%d\n",
            baseline.time_total, baseline.genetic_gain, baseline.genetic_var, n_sel)

    for r in ranks
        print("  rank $r: ")
        for rep in 1:n_replicates
            sol  = solve_rsvd(G, g_vec; rank=r, gamma=gamma)
            cmp  = compare(baseline, sol, G)
            n_sel_approx = sum(sol.contributions .> SEL_THRESH)

            push!(raw_rows, (
                gamma          = gamma,
                rank           = r,
                replicate      = rep,
                n_sel_baseline = n_sel,
                n_sel_approx   = n_sel_approx,
                gain_diff_pct  = cmp.gain_diff_pct,
                var_diff_pct   = cmp.var_diff_pct,
                speedup        = cmp.speedup,
                spearman_rho   = cmp.spearman_rho,
                overlap_pct    = cmp.overlap_pct,
                time_baseline  = baseline.time_total,
                time_rsvd      = sol.time_total,
                time_decomp    = sol.time_decomp,
                time_opt       = sol.time_opt,
                status         = sol.status
            ))
            print(".")
        end
        rep_rows = filter(row -> row.gamma == gamma && row.rank == r, raw_rows)
        @printf("  gain_diff=%.3f%%  speedup=%.1fx\n",
                mean(row.gain_diff_pct for row in rep_rows),
                mean(row.speedup       for row in rep_rows))
    end
    println()
end

# =============================================================================
# 4. SUMMARY ACROSS REPLICATES
# =============================================================================

raw_df = DataFrame(raw_rows)

summary_df = combine(
    groupby(raw_df, [:gamma, :rank]),
    :n_sel_baseline => first        => :n_sel_baseline,
    :gain_diff_pct  => mean         => :mean_gain_diff_pct,
    :gain_diff_pct  => std          => :sd_gain_diff_pct,
    :var_diff_pct   => mean         => :mean_var_diff_pct,
    :speedup        => mean         => :mean_speedup,
    :speedup        => std          => :sd_speedup,
    :spearman_rho   => mean         => :mean_spearman_rho,
    :overlap_pct    => mean         => :mean_overlap_pct,
    :time_baseline  => first        => :time_baseline,
    :time_rsvd      => mean         => :mean_time_rsvd,
    :time_decomp    => mean         => :mean_time_decomp,
    :time_opt       => mean         => :mean_time_opt
)

# =============================================================================
# 5. ELBOW TABLE -- recommended rank per gamma
# =============================================================================

println("="^70)
println("Elbow table -- recommended rank per gamma (gain_diff < $(GAIN_THRESH)%)")
println("="^70)
println()
println(rpad("gamma", 10), rpad("n_sel", 8), rpad("rec_rank", 10),
        rpad("speedup", 10), rpad("gain_diff%", 12), rpad("spearman_rho", 14), "overlap%")
println("-"^74)

elbow_rows = []

for gamma in gammas
    sub = filter(r -> r.gamma == gamma, summary_df)
    isempty(sub) && continue
    sort!(sub, :rank)

    good = filter(r -> r.mean_gain_diff_pct < GAIN_THRESH, eachrow(sub))
    if isempty(good)
        rec_rank    = maximum(sub.rank)
        rec_row     = filter(r -> r.rank == rec_rank, eachrow(sub))[1]
        flag        = ">"
    else
        rec_row     = first(good)
        rec_rank    = rec_row.rank
        flag        = " "
    end

    n_sel = rec_row.n_sel_baseline
    @printf("%s%-9.4g %-8d %-10d %-10.2f %-12.4f %-14.4f %.1f\n",
            flag, gamma, n_sel, rec_rank,
            rec_row.mean_speedup, rec_row.mean_gain_diff_pct,
            rec_row.mean_spearman_rho, rec_row.mean_overlap_pct)

    push!(elbow_rows, (
        gamma           = gamma,
        n_sel_baseline  = n_sel,
        rec_rank        = rec_rank,
        below_threshold = isempty(good) ? false : true,
        mean_speedup    = rec_row.mean_speedup,
        sd_speedup      = rec_row.sd_speedup,
        mean_gain_diff  = rec_row.mean_gain_diff_pct,
        sd_gain_diff    = rec_row.sd_gain_diff_pct,
        mean_spearman   = rec_row.mean_spearman_rho,
        mean_overlap    = rec_row.mean_overlap_pct,
        time_baseline   = rec_row.time_baseline,
        mean_time_rsvd  = rec_row.mean_time_rsvd
    ))
end

println()
println("  (>) indicates no rank achieved <$(GAIN_THRESH)% gain diff; showing max rank instead.")
println()

elbow_df = DataFrame(elbow_rows)

# =============================================================================
# 6. EXPORT
# =============================================================================

mkpath(OUT_DIR)

raw_csv     = joinpath(OUT_DIR, "gamma_rank_sweep_raw_spruce.csv")
summary_csv = joinpath(OUT_DIR, "gamma_rank_sweep_summary_spruce.csv")
elbow_csv   = joinpath(OUT_DIR, "gamma_rank_sweep_elbow_spruce.csv")

CSV.write(raw_csv,     raw_df)
CSV.write(summary_csv, summary_df)
CSV.write(elbow_csv,   elbow_df)

@printf("Saved raw results    : %s\n", raw_csv)
@printf("Saved summary table  : %s\n", summary_csv)
@printf("Saved elbow table    : %s\n", elbow_csv)

println()
println("="^70)
println("SWEEP COMPLETE  --  $(Dates.now())")
println("="^70)

end # if !RUN_PILOT
