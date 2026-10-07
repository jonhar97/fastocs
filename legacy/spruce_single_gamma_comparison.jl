# =============================================================================
# Norway Spruce -- Single-Gamma (gamma=300) Rank Selection & Method Comparison
#
# Replaces the six-file comprehensive_ocs_comparison_fixed.jl + patch-file
# pipeline (compare_solutions_rank30.jl, rank_selection_implementation.jl,
# rank_selection_with_uncertainty.jl, compare_pca_methods_timing.jl,
# timing_statistics_extension.jl) that produced the manuscript's original
# Table 1 ("rank_compact"), Table "solution_quality", Figure 1
# ("rank_selection"), and Figure 2 ("solution_concordance").
#
# CONFIRMED FIX vs. the old pipeline:
#   comprehensive_ocs_comparison_fixed.jl's load_and_create_index() used
#   traits = Dict("Hjd17"=>1.0, "Htv17"=>1.0, "Sprant17"=>-1.0, "Lev17"=>1.0)
#   and divided by 4.0 -- a 4-trait index that includes Lev17. Every other
#   analysis in the manuscript now uses the 3-trait index agreed with Jon,
#   (Hjd17 + Htv17 - Sprant17)/3, with Lev17 excluded. Fixed below.
#
# OTHER CHANGES vs. the old pipeline:
#   - Constraint scale unified to sum(c) = 1 (was sum(c) = N_select = 100),
#     matching gamma_rank_sweep_spruce.jl, so gamma is comparable across the
#     whole spruce section of the manuscript.
#   - gamma fixed at 300 (not the old default 50), matching the chosen
#     operating point in the new gamma x rank sweep, so this section and
#     that one describe the same OCS scenario.
#   - ID alignment between the H-matrix and EBV files is verified via an ID
#     lookup + diagonal sanity check (see gamma_rank_sweep_spruce.jl),
#     rather than assumed from row order.
#   - The full deterministic SVD (for "PCA Standard") is computed ONCE,
#     outside the rank loop, and truncated per rank. LAPACK's svd() always
#     computes the complete decomposition regardless of target rank, so
#     recomputing it per rank/replicate (as the old benchmark_method()
#     pattern effectively would) would mean up to 100 redundant full
#     5,525 x 5,525 SVDs -- extremely slow, and it also misrepresents the
#     real cost structure: unlike RSVD, standard truncated SVD has no
#     "compute only the top k" shortcut, so its cost is fixed regardless of
#     rank. This is also the mechanistic reason the manuscript's abstract
#     says standard SVD "only marginally reduced computing time" versus
#     RSVD -- worth keeping in mind when writing up these results.
#   - Factor Analysis (Method 2 in the old file) is dropped -- it appears in
#     the old code but not in any current manuscript figure or table.
#
# Outputs:
#   spruce_rank_comparison_raw.csv      -- one row per method x rank x replicate
#   spruce_rank_comparison_summary.csv  -- mean/sd/CV per method x rank
#   spruce_concordance_rank30.csv       -- per-individual contributions and
#                                          ranks for baseline/PCA
#                                          Standard/PCA Randomized at the
#                                          reference rank, for Figure 2 and
#                                          the solution-quality table
#
# Once you've run this and shared the three CSVs plus console output, I'll
# build the Figure 1 and Figure 2 analogs the same way we did for the
# gamma x rank sweep.
# =============================================================================

using LinearAlgebra
using Random
using Statistics
using StatsBase   # for corkendall (Kendall's tau) -- `] add StatsBase` if not installed
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
OUT_DIR  = "EDIT_ME/NorwaySpruceData/results_single_gamma/"

gebv_file_hjd17    = joinpath(BASE_DIR, "Save/MCMC_H55253/EBV_Hjd17.txt")
gebv_file_htv17    = joinpath(BASE_DIR, "Save/MCMC_H55253/EBV_Htv17.txt")
gebv_file_sprant17 = joinpath(BASE_DIR, "Save/MCMC_H55253/EBV_Sprant17.txt")
geno_file          = joinpath(BASE_DIR, "Save/Hmat_5525_spruce_tau_1_omega_1_PDF.txt")

INDEX_WEIGHTS = (hjd17 = 1/3, htv17 = 1/3, sprant17 = -1/3)

GAMMA          = 300.0     # matches the chosen operating point in the gamma x rank sweep
SEL_THRESH     = 1e-4
REFERENCE_RANK = 30        # rank at which per-individual concordance is reported
ranks          = [5, 10, 15, 20, 25, 30, 40, 50, 75, 100]
n_reps         = 10        # timing/stochastic replicates per (method, rank)

# =============================================================================
# 1. DATA LOADING (same approach as gamma_rank_sweep_spruce.jl)
# =============================================================================

println("="^70)
println("Norway Spruce -- Single-Gamma (gamma=$GAMMA) Rank & Method Comparison")
println("="^70)
println("Timestamp: $(Dates.now())")
println()

println("[1] Loading EBV files and building selection index...")
hjd_df    = CSV.read(gebv_file_hjd17,    DataFrame; delim=',')
htv_df    = CSV.read(gebv_file_htv17,    DataFrame; delim=',')
sprant_df = CSV.read(gebv_file_sprant17, DataFrame; delim=',')

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

idx_df.index = INDEX_WEIGHTS.hjd17    .* idx_df.EBV_hjd17 .+
               INDEX_WEIGHTS.htv17    .* idx_df.EBV_htv17 .+
               INDEX_WEIGHTS.sprant17 .* idx_df.EBV_sprant17

id_to_g = Dict(Int(idx_df.ID[i]) => idx_df.index[i] for i in eachindex(idx_df.ID))
@printf("  n=%d individuals with a complete 3-trait index (pre-alignment)\n\n", nrow(idx_df))

println("[2] Building relationship matrix and aligning IDs...")
M = readdlm(geno_file, ',', header=false)
all_ids = Int.(round.(M[:, 1]))
n_total = length(all_ids)
G       = Matrix(M[:, 2:end])
@assert size(G, 1) == size(G, 2) == n_total "geno_file is not square after dropping the ID column."
@assert all(abs.(diag(G) .- 1.0) .< 0.3) "geno_file diagonal is not close to 1.0 -- row/column order mismatch."
G = (G + G') / 2

missing_ids = [id for id in all_ids if !haskey(id_to_g, id)]
if !isempty(missing_ids)
    @warn "$(length(missing_ids)) individuals in geno_file have no 3-trait index -- excluding them."
end
keep_mask = [haskey(id_to_g, id) for id in all_ids]
all_ids   = all_ids[keep_mask]
G         = G[keep_mask, keep_mask]
n_total   = length(all_ids)
g_vec     = [id_to_g[id] for id in all_ids]

@printf("  Aligned relationship matrix: %d x %d (diag mean=%.4f)\n\n", n_total, n_total, mean(diag(G)))

# =============================================================================
# 2. METHODS
# =============================================================================

function randomized_svd(A::Matrix{T}, k::Int; p::Int=10, q::Int=2) where T
    n, m = size(A)
    l = min(k + p, min(n, m))
    Ω = randn(T, m, l)
    Y = A * Ω
    for _ in 1:q
        Y = A * (A' * Y)
    end
    Q, _ = qr(Y)
    Q = Matrix(Q)
    B = Q' * A
    Ũ, S, V = svd(B)
    U = Q * Ũ
    return U[:, 1:k], S[1:k], V[:, 1:k]
end

function solve_full_dense(G, g; gamma)
    n = length(g)
    t = @elapsed begin
        model = Model(optimizer_with_attributes(OSQP.Optimizer,
            "max_iter" => 20000, "eps_abs" => 1e-6, "eps_rel" => 1e-6, "verbose" => false))
        @variable(model, x[1:n] >= 0)
        @objective(model, Min, 0.5 * (x' * G * x) - (1/(2*gamma)) * dot(g, x))
        @constraint(model, sum(x) == 1.0)
        JuMP.optimize!(model)
        c = JuMP.value.(x)
    end
    return (contributions=c, status=string(termination_status(model)),
            genetic_gain=dot(c, g), genetic_var=c'*G*c, time_total=t)
end

"""Solve OCS from a precomputed low-rank factor F = U*sqrt(S) (n x rank).
Only the OSQP solve is timed here -- decomposition timing is handled
separately by the caller (see Section 4: shared once for PCA Standard,
per-replicate for PCA Randomized)."""
function solve_from_factor(G, g, F; gamma)
    n, rank = size(F)
    D_vals = diag(G) - vec(sum(F.^2, dims=2))
    t_opt = @elapsed begin
        model = Model(optimizer_with_attributes(OSQP.Optimizer,
            "max_iter" => 20000, "eps_abs" => 1e-6, "eps_rel" => 1e-6, "verbose" => false))
        @variable(model, x[1:n] >= 0)
        @variable(model, y[1:rank])
        @objective(model, Min,
            0.5 * sum(D_vals[i] * x[i]^2 for i in 1:n) + 0.5 * dot(y, y) - (1/(2*gamma)) * dot(g, x))
        @constraint(model, sum(x) == 1.0)
        for j in 1:rank
            @constraint(model, y[j] == sum(F[i, j] * x[i] for i in 1:n))
        end
        JuMP.optimize!(model)
        c = JuMP.value.(x)
    end
    return (contributions=c, status=string(termination_status(model)),
            genetic_gain=dot(c, g), genetic_var=c'*G*c, time_opt=t_opt)
end

function compare(baseline, approx_contributions, approx_gain, approx_var, approx_time_total)
    gain_diff_pct = abs(approx_gain - baseline.genetic_gain) / abs(baseline.genetic_gain) * 100
    var_diff_pct  = abs(approx_var  - baseline.genetic_var)  / abs(baseline.genetic_var)  * 100
    speedup       = baseline.time_total / approx_time_total
    rb = sortperm(sortperm(baseline.contributions))
    ra = sortperm(sortperm(approx_contributions))
    spearman_rho  = cor(Float64.(rb), Float64.(ra))
    kendall_tau   = corkendall(baseline.contributions, approx_contributions)
    sel_b = findall(baseline.contributions .> SEL_THRESH)
    sel_a = findall(approx_contributions   .> SEL_THRESH)
    overlap = length(intersect(sel_b, sel_a)) / max(length(sel_b), 1) * 100
    return (gain_diff_pct=gain_diff_pct, var_diff_pct=var_diff_pct, speedup=speedup,
            spearman_rho=spearman_rho, kendall_tau=kendall_tau, overlap_pct=overlap)
end

# =============================================================================
# 3. BASELINE (deterministic, solved once and reused throughout)
# =============================================================================

println("[3] Full-dense baseline (gamma=$GAMMA)...")
baseline = solve_full_dense(G, g_vec; gamma=GAMMA)
n_sel_baseline = sum(baseline.contributions .> SEL_THRESH)
@printf("  status=%s  time=%.1fs  gain=%.4f  n_selected=%d (%.2f%% of n=%d)\n\n",
        baseline.status, baseline.time_total, baseline.genetic_gain,
        n_sel_baseline, 100*n_sel_baseline/n_total, n_total)

# =============================================================================
# 4. FULL DETERMINISTIC SVD (computed ONCE -- see header note on why)
# =============================================================================

println("[4] Computing full deterministic SVD once (reused for all PCA Standard ranks)...")
t_full_svd = @elapsed begin
    Fsvd = svd(G)
end
@printf("  Full SVD time: %.1fs (this cost is fixed regardless of target rank)\n\n", t_full_svd)

# =============================================================================
# 5. RANK x METHOD x REPLICATE SWEEP
# =============================================================================

println("[5] Rank sweep: PCA Standard vs PCA Randomized, n_reps=$n_reps per rank")
rows = []
for r in ranks
    print("  rank $r: ")

    # --- PCA Standard: decomposition is the shared t_full_svd; only the
    #     OSQP solve is re-timed per replicate (deterministic problem, so
    #     replicates capture solve-time noise, not solution variability).
    U_std, S_std = Fsvd.U[:, 1:r], Fsvd.S[1:r]
    F_std = U_std * Diagonal(sqrt.(S_std))
    for rep in 1:n_reps
        sol = solve_from_factor(G, g_vec, F_std; gamma=GAMMA)
        cmp = compare(baseline, sol.contributions, sol.genetic_gain, sol.genetic_var, t_full_svd + sol.time_opt)
        push!(rows, (method="PCA Standard", rank=r, replicate=rep, n_sel_baseline=n_sel_baseline,
                      gain_diff_pct=cmp.gain_diff_pct, var_diff_pct=cmp.var_diff_pct, speedup=cmp.speedup,
                      spearman_rho=cmp.spearman_rho, kendall_tau=cmp.kendall_tau, overlap_pct=cmp.overlap_pct,
                      time_baseline=baseline.time_total, time_decomp=t_full_svd, time_opt=sol.time_opt,
                      time_total=t_full_svd + sol.time_opt, status=sol.status))
        print(".")
    end

    # --- PCA Randomized: decomposition genuinely differs each replicate
    #     (fresh random projection), so both decomp and opt are re-timed.
    for rep in 1:n_reps
        t_decomp = @elapsed begin
            U_r, S_r, _ = randomized_svd(G, r; p=10, q=2)
        end
        F_r = U_r * Diagonal(sqrt.(S_r))
        sol = solve_from_factor(G, g_vec, F_r; gamma=GAMMA)
        cmp = compare(baseline, sol.contributions, sol.genetic_gain, sol.genetic_var, t_decomp + sol.time_opt)
        push!(rows, (method="PCA Randomized", rank=r, replicate=rep, n_sel_baseline=n_sel_baseline,
                      gain_diff_pct=cmp.gain_diff_pct, var_diff_pct=cmp.var_diff_pct, speedup=cmp.speedup,
                      spearman_rho=cmp.spearman_rho, kendall_tau=cmp.kendall_tau, overlap_pct=cmp.overlap_pct,
                      time_baseline=baseline.time_total, time_decomp=t_decomp, time_opt=sol.time_opt,
                      time_total=t_decomp + sol.time_opt, status=sol.status))
        print(".")
    end
    println()
end
raw_df = DataFrame(rows)

summary_df = combine(groupby(raw_df, [:method, :rank]),
    :n_sel_baseline => first => :n_sel_baseline,
    :gain_diff_pct  => mean  => :mean_gain_diff_pct,  :gain_diff_pct => std => :sd_gain_diff_pct,
    :var_diff_pct   => mean  => :mean_var_diff_pct,
    :speedup        => mean  => :mean_speedup,        :speedup       => std => :sd_speedup,
    :spearman_rho   => mean  => :mean_spearman_rho,
    :kendall_tau    => mean  => :mean_kendall_tau,
    :overlap_pct    => mean  => :mean_overlap_pct,
    :time_baseline  => first => :time_baseline,
    :time_total     => mean  => :mean_time_total,     :time_total    => std => :sd_time_total,
    :time_decomp    => mean  => :mean_time_decomp,
    :time_opt       => mean  => :mean_time_opt
)
summary_df.cv_time_pct = 100 .* summary_df.sd_time_total ./ summary_df.mean_time_total

# =============================================================================
# 6. PER-INDIVIDUAL CONCORDANCE AT THE REFERENCE RANK (Figure 2 / solution_quality)
# =============================================================================

println("\n[6] Per-individual concordance at rank $REFERENCE_RANK...")
U_ref, S_ref = Fsvd.U[:, 1:REFERENCE_RANK], Fsvd.S[1:REFERENCE_RANK]
F_ref_std = U_ref * Diagonal(sqrt.(S_ref))
sol_std_ref  = solve_from_factor(G, g_vec, F_ref_std; gamma=GAMMA)

U_r_ref, S_r_ref, _ = randomized_svd(G, REFERENCE_RANK; p=10, q=2)
F_ref_rsvd = U_r_ref * Diagonal(sqrt.(S_r_ref))
sol_rsvd_ref = solve_from_factor(G, g_vec, F_ref_rsvd; gamma=GAMMA)

rank_desc(x) = sortperm(sortperm(-x))   # rank 1 = largest contribution

concordance_df = DataFrame(
    id                            = all_ids,
    baseline_contribution         = baseline.contributions,
    pca_standard_contribution     = sol_std_ref.contributions,
    pca_randomized_contribution   = sol_rsvd_ref.contributions,
    baseline_rank                 = rank_desc(baseline.contributions),
    pca_standard_rank             = rank_desc(sol_std_ref.contributions),
    pca_randomized_rank           = rank_desc(sol_rsvd_ref.contributions)
)

cmp_std_ref  = compare(baseline, sol_std_ref.contributions,  sol_std_ref.genetic_gain,  sol_std_ref.genetic_var,  t_full_svd)
cmp_rsvd_ref = compare(baseline, sol_rsvd_ref.contributions, sol_rsvd_ref.genetic_gain, sol_rsvd_ref.genetic_var, 1.0)
cmp_std_vs_rsvd = compare(
    (contributions=sol_std_ref.contributions, genetic_gain=sol_std_ref.genetic_gain,
     genetic_var=sol_std_ref.genetic_var, time_total=1.0),
    sol_rsvd_ref.contributions, sol_rsvd_ref.genetic_gain, sol_rsvd_ref.genetic_var, 1.0
)

@printf("  PCA Standard   vs Full Dense: Spearman=%.3f  Kendall=%.3f  overlap=%.1f%%\n",
        cmp_std_ref.spearman_rho, cmp_std_ref.kendall_tau, cmp_std_ref.overlap_pct)
@printf("  PCA Randomized vs Full Dense: Spearman=%.3f  Kendall=%.3f  overlap=%.1f%%\n",
        cmp_rsvd_ref.spearman_rho, cmp_rsvd_ref.kendall_tau, cmp_rsvd_ref.overlap_pct)
@printf("  PCA Randomized vs PCA Standard: Spearman=%.3f  Kendall=%.3f  overlap=%.1f%%\n\n",
        cmp_std_vs_rsvd.spearman_rho, cmp_std_vs_rsvd.kendall_tau, cmp_std_vs_rsvd.overlap_pct)

# =============================================================================
# 7. EXPORT
# =============================================================================

mkpath(OUT_DIR)
raw_csv         = joinpath(OUT_DIR, "spruce_rank_comparison_raw.csv")
summary_csv     = joinpath(OUT_DIR, "spruce_rank_comparison_summary.csv")
concordance_csv = joinpath(OUT_DIR, "spruce_concordance_rank$(REFERENCE_RANK).csv")

CSV.write(raw_csv, raw_df)
CSV.write(summary_csv, summary_df)
CSV.write(concordance_csv, concordance_df)

println("Saved:")
println("  ", raw_csv)
println("  ", summary_csv)
println("  ", concordance_csv)
println()
println("="^70)
println("COMPLETE -- ", Dates.now())
println("="^70)
