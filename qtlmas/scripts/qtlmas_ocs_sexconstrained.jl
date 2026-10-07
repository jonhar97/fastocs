# =============================================================================
# QTLMAS 2010 — Sex-Constrained OCS with RSVD Low-Rank Approximation
#
# Implements Optimum Contribution Selection (OCS) on the QTL-MAS 2010 dataset
# using GEBVs from GBLUP/AI-REML and the genomic relationship matrix G.
#
# Sex constraints:
#   The contribution vector x is partitioned into males (x_m) and females (x_f).
#   Separate equality constraints enforce:
#       sum(x_m) == N_select / 2   (sire contributions)
#       sum(x_f) == N_select / 2   (dam contributions)
#   This ensures balanced genomic representation across sexes, which is the
#   standard OCS formulation for livestock/plant populations with distinct sexes.
#   (Meuwissen 1997; Woolliams et al. 2015)
#
# Methods compared:
#   1. Full Dense (baseline)
#   2. PCA Standard SVD (auxiliary variable formulation)
#   3. PCA Randomized SVD (RSVD — the COSMO OC method)
#
# INPUT FILES (adjust BASE_DIR):
#   tbv.txt          — comma-delimited; col1=ID, col2=sex(0/1), col3=phenotype ...
#   GEBV_output.txt  — tab-delimited with header; cols: ID, GEBV, PEV_SE, ...
#   QTLMAS2010gen/QTLMAS2010gen.txt — comma-delimited genotype matrix (0/1/2)
#
# OUTPUT FILES:
#   ocs_results_sexconstrained.csv  — rank comparison summary table
#   ocs_contributions_full.csv      — individual contributions from full-dense run
#   ocs_contributions_rsvd.csv      — individual contributions from best RSVD rank
# =============================================================================

using LinearAlgebra
using Random
using Statistics
using Printf
using OSQP
using JuMP
using SparseArrays
using DelimitedFiles
using DataFrames
using CSV
using Dates
using JLD2

# =============================================================================
# 0. CONFIGURATION
# =============================================================================

BASE_DIR  = normpath(joinpath(@__DIR__, "..", "data"))
OUT_DIR   = normpath(joinpath(@__DIR__, "..", "output"))

tbv_file  = joinpath(BASE_DIR, "tbv.txt")
gebv_file = joinpath(BASE_DIR, "GEBV_output.txt")
geno_file = joinpath(BASE_DIR, "QTLMAS2010gen/QTLMAS2010gen.txt")

# OCS parameters
N_select       = 100        # Total number of selected individuals (split evenly by sex)
gamma_default  = 1.0       # Constraint on rate of inbreeding (higher = more gain, less constraint)
ranks_to_test  = [5, 10, 15, 20, 25, 30, 40, 50, 75, 100]   # Extended range to find convergence
n_replicates   = 5          # RSVD replicates per rank (captures randomness)

# Spectrum diagnostic: how many singular values to compute for the explained-variance plot.
# Full SVD of a 3226×3226 matrix is expensive (~minutes); we cap at this value.
# Set to n_total for a complete spectrum (slow) or a smaller number for a fast diagnostic.
spectrum_rank  = 500        # top-K singular values to inspect

# =============================================================================
# 1. DATA LOADING
# =============================================================================

println("="^70)
println("QTLMAS 2010 — Sex-Constrained OCS (RSVD Low-Rank)")
println("="^70)
println("Timestamp: $(Dates.now())")
println()

# --- TBV / sex file -----------------------------------------------------------
println("[1] Loading data...")
tbv_raw  = readdlm(tbv_file, ',', header=false)
all_ids  = Int.(tbv_raw[:, 1])
all_sex  = Int.(tbv_raw[:, 2])   # 0 = female, 1 = male (matches GBLUP script)
n_total  = length(all_ids)

male_idx   = findall(all_sex .== 1)
female_idx = findall(all_sex .== 0)
n_males    = length(male_idx)
n_females  = length(female_idx)

@printf("  Total individuals : %d\n", n_total)
@printf("  Males (sex=1)     : %d\n", n_males)
@printf("  Females (sex=0)   : %d\n", n_females)

# --- GEBVs -------------------------------------------------------------------
gebv_df  = CSV.read(gebv_file, DataFrame; delim='\t')
gebv_ids = Int.(gebv_df.ID)
gebv_vals = Float64.(gebv_df.GEBV)

# Align GEBVs to master ID order from tbv.txt
id_to_gebv = Dict(gebv_ids[i] => gebv_vals[i] for i in eachindex(gebv_ids))
g_vec = [get(id_to_gebv, id, NaN) for id in all_ids]

n_missing_gebv = sum(isnan.(g_vec))
if n_missing_gebv > 0
    @warn "  $n_missing_gebv individuals in tbv.txt have no GEBV — setting to 0"
    g_vec[isnan.(g_vec)] .= 0.0
end
println("  GEBVs aligned. Mean GEBV = $(@sprintf("%.4f", mean(g_vec)))")

# --- Genotype matrix & GRM ---------------------------------------------------
println("  Building GRM (VanRaden 2008) — this may take a moment...")
M = readdlm(geno_file, ',', header=false)
@assert size(M, 1) == n_total "Genotype matrix row count mismatch!"

n_markers = size(M, 2)
N_mat = M .- 1                              # centre to {-1, 0, 1}
p_freq = sum(M; dims=1) ./ (n_total * 2)   # allele frequencies
Z = N_mat .- 2 .* (p_freq .- 0.5)          # centred marker matrix
denom = 2 * sum(p_freq .* (1 .- p_freq))
G = (Z * Z') ./ denom

@printf("  GRM built: %d × %d  (diag mean = %.4f)\n", n_total, n_total, mean(diag(G)))

# Force symmetry (guard against floating-point asymmetry)
G = (G + G') / 2

println()

# =============================================================================
# 2. SPECTRAL DIAGNOSTIC
# =============================================================================
# Before running OCS, inspect how many singular values are needed to capture
# most of G's variance. This tells us the intrinsic rank of the GRM and
# explains why low-rank approximations may or may not work well.
#
# We use RSVD (fast) to estimate the top-K singular values rather than a full
# SVD, which would take O(n³) time for n=3226.
#
# Key quantities reported:
#   - Cumulative explained variance at ranks in ranks_to_test
#   - Rank needed to reach 90%, 95%, 99% of total variance
#   - Effective rank (entropy-based): exp(-Σ p_i log p_i) where p_i = s_i/Σs_j
#     This is a single-number summary of spectral spread (low = concentrated,
#     high = diffuse / needs many components).

println("="^70)
println("Spectral Diagnostic — G matrix singular value spectrum")
println("="^70)
println("Computing top-$spectrum_rank singular values via RSVD...")

# Use more power iterations here for accurate singular value estimates
spec_time = @elapsed begin
    # We only need singular values (not vectors) for the diagnostic.
    # RSVD returns U, S, V — we use only S.
    function rsvd_values_only(A::Matrix{T}, k::Int; p::Int=20, q::Int=3) where T
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
        _, S, _ = svd(B)
        return S[1:k]
    end

    S_spec = rsvd_values_only(G, spectrum_rank; p=20, q=3)
end

# Total variance approximated from the top-K values.
# Note: true total = tr(G), but since we only have top-K eigenvalues from RSVD,
# we use tr(G) as the denominator for cumulative explained variance.
tr_G      = tr(G)
cumvar    = cumsum(S_spec) ./ tr_G .* 100   # cumulative % of tr(G) explained

@printf("  Done in %.1f s\n\n", spec_time)
@printf("  Total variance (tr G)          : %.4f\n", tr_G)
@printf("  Top-%d singular values capture : %.2f%% of tr(G)\n\n", spectrum_rank, cumvar[end])

# Cumulative variance at OCS ranks of interest
println("  Cumulative explained variance at selected ranks:")
println("  " * rpad("Rank", 8) * rpad("Cum. var (%)", 16) * "Singular value")
println("  " * "-"^40)
check_ranks = sort(unique(vcat(ranks_to_test, [10, 25, 50, 100, 200, 300, 500])))
for r in check_ranks
    r > spectrum_rank && continue
    @printf("  %-8d %-16.2f %.6f\n", r, cumvar[r], S_spec[r])
end

# Thresholds: rank needed to reach 90 / 95 / 99 %
for thr in [90.0, 95.0, 99.0]
    idx = findfirst(cumvar .>= thr)
    if isnothing(idx)
        @printf("\n  Rank for %.0f%% variance : > %d (not reached in top-%d)\n", thr, spectrum_rank, spectrum_rank)
    else
        @printf("\n  Rank for %.0f%% variance : %d  (singular value = %.6f)\n", thr, idx, S_spec[idx])
    end
end

# Effective rank (entropy-based)
p_spec     = S_spec ./ sum(S_spec)
eff_rank   = exp(-sum(p_i * log(p_i) for p_i in p_spec if p_i > 0))
@printf("\n  Effective rank (entropy, top-%d): %.1f\n", spectrum_rank, eff_rank)

# Elbow: rank where marginal gain in cumvar drops below 0.1%
marginal   = diff(cumvar)
elbow_idx  = findfirst(marginal .< 0.1)
if !isnothing(elbow_idx)
    @printf("  Spectral elbow (Δcumvar < 0.1%%) : rank %d\n", elbow_idx)
end

println()

# Export spectrum to CSV for plotting in R
mkpath(OUT_DIR)
spec_df = DataFrame(
    rank       = 1:spectrum_rank,
    sing_value = S_spec,
    cumvar_pct = cumvar
)
spec_csv = joinpath(OUT_DIR, "grm_singular_value_spectrum.csv")
CSV.write(spec_csv, spec_df)
println("  Spectrum saved: $spec_csv")
println()

# =============================================================================
# 3. RANDOMIZED SVD (solver utility)
# =============================================================================

"""
Randomized SVD for fast low-rank approximation of a symmetric PSD matrix.
Returns U, S, V such that A ≈ U * Diagonal(S) * V'.
"""
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

# =============================================================================
# 4. OCS SOLVERS — sex-constrained formulation
# =============================================================================
#
# Standard OCS objective (portfolio formulation):
#   min  0.5 * x' G x  -  (1 / 2γ) * g' x
#   s.t. sum(x[male_idx])   == N_select / 2    (sire contributions)
#        sum(x[female_idx]) == N_select / 2    (dam contributions)
#        x >= 0
#
# Low-rank (auxiliary variable) reformulation:
#   G ≈ F F' + diag(D)   where F = U * sqrt(S),  n × k
#
#   min  0.5 * (x' diag(D) x  +  ||y||²)  -  (1 / 2γ) * g' x
#   s.t. y[j] = sum_i F[i,j] * x[i]   ∀ j       (link constraints)
#        sum(x[male_idx])   == N_select / 2
#        sum(x[female_idx]) == N_select / 2
#        x >= 0
#
# Both formulations yield the same optimal x when G = F F' + diag(D) exactly.

"""
Full Dense OCS with sex constraints (baseline).
"""
function ocs_full_dense_sex(G::Matrix, g::Vector,
                             male_idx::Vector{Int}, female_idx::Vector{Int},
                             N_select::Int;
                             gamma::Float64=50.0, verbose::Bool=false)

    n = length(g)
    N_m = N_select ÷ 2          # sire quota
    N_f = N_select - N_m        # dam quota

    verbose && println("  [Full Dense] n=$n, N_m=$N_m, N_f=$N_f, γ=$gamma")

    time_total = @elapsed begin
        model = Model(
            optimizer_with_attributes(OSQP.Optimizer,
                "max_iter" => 20000,
                "eps_abs"  => 1e-6,
                "eps_rel"  => 1e-6,
                "verbose"  => false)
        )

        @variable(model, x[1:n] >= 0)
        @objective(model, Min,
            0.5 * (x' * G * x) - (1 / (2*gamma)) * dot(g, x)
        )
        # Sex constraints: contributions sum to sex-specific quotas
        @constraint(model, sum(x[i] for i in male_idx)   == Float64(N_m))
        @constraint(model, sum(x[i] for i in female_idx) == Float64(N_f))

        JuMP.optimize!(model)
        c = JuMP.value.(x)
    end

    status          = termination_status(model)
    genetic_gain    = dot(c, g)
    genetic_variance = c' * G * c

    return (
        contributions    = c,
        time_total       = time_total,
        time_decomp      = 0.0,
        time_opt         = time_total,
        status           = status,
        genetic_gain     = genetic_gain,
        genetic_variance = genetic_variance,
        method           = "Full Dense"
    )
end

"""
PCA Standard SVD OCS with sex constraints and auxiliary variable formulation.
"""
function ocs_pca_standard_sex(G::Matrix, g::Vector,
                               male_idx::Vector{Int}, female_idx::Vector{Int},
                               N_select::Int;
                               rank::Int=50, gamma::Float64=50.0, verbose::Bool=false)

    n = length(g)
    N_m = N_select ÷ 2
    N_f = N_select - N_m

    time_decomp = @elapsed begin
        F_svd = svd(G)
        U = F_svd.U[:, 1:rank]
        S = F_svd.S[1:rank]
        F = U * Diagonal(sqrt.(S))          # n × rank factor matrix
        H_approx_diag = vec(sum(F.^2, dims=2))
        D_vals = diag(G) - H_approx_diag
    end

    time_opt = @elapsed begin
        model = Model(
            optimizer_with_attributes(OSQP.Optimizer,
                "max_iter" => 20000,
                "eps_abs"  => 1e-6,
                "eps_rel"  => 1e-6,
                "verbose"  => false)
        )

        @variable(model, x[1:n] >= 0)
        @variable(model, y[1:rank])

        @objective(model, Min,
            0.5 * sum(D_vals[i] * x[i]^2 for i in 1:n) +
            0.5 * dot(y, y) -
            (1 / (2*gamma)) * dot(g, x)
        )

        @constraint(model, sum(x[i] for i in male_idx)   == Float64(N_m))
        @constraint(model, sum(x[i] for i in female_idx) == Float64(N_f))

        for j in 1:rank
            @constraint(model, y[j] == sum(F[i, j] * x[i] for i in 1:n))
        end

        JuMP.optimize!(model)
        c = JuMP.value.(x)
    end

    status           = termination_status(model)
    genetic_gain     = dot(c, g)
    genetic_variance = c' * G * c

    return (
        contributions    = c,
        time_decomp      = time_decomp,
        time_opt         = time_opt,
        time_total       = time_decomp + time_opt,
        status           = status,
        genetic_gain     = genetic_gain,
        genetic_variance = genetic_variance,
        rank             = rank,
        method           = "PCA Standard"
    )
end

"""
PCA Randomized SVD OCS with sex constraints (COSMO OC method).
"""
function ocs_pca_randomized_sex(G::Matrix, g::Vector,
                                 male_idx::Vector{Int}, female_idx::Vector{Int},
                                 N_select::Int;
                                 rank::Int=50, gamma::Float64=50.0, verbose::Bool=false)

    n = length(g)
    N_m = N_select ÷ 2
    N_f = N_select - N_m

    time_decomp = @elapsed begin
        U, S, _ = randomized_svd(G, rank; p=10, q=2)
        F = U * Diagonal(sqrt.(S))
        H_approx_diag = vec(sum(F.^2, dims=2))
        D_vals = diag(G) - H_approx_diag
    end

    time_opt = @elapsed begin
        model = Model(
            optimizer_with_attributes(OSQP.Optimizer,
                "max_iter" => 20000,
                "eps_abs"  => 1e-6,
                "eps_rel"  => 1e-6,
                "verbose"  => false)
        )

        @variable(model, x[1:n] >= 0)
        @variable(model, y[1:rank])

        @objective(model, Min,
            0.5 * sum(D_vals[i] * x[i]^2 for i in 1:n) +
            0.5 * dot(y, y) -
            (1 / (2*gamma)) * dot(g, x)
        )

        @constraint(model, sum(x[i] for i in male_idx)   == Float64(N_m))
        @constraint(model, sum(x[i] for i in female_idx) == Float64(N_f))

        for j in 1:rank
            @constraint(model, y[j] == sum(F[i, j] * x[i] for i in 1:n))
        end

        JuMP.optimize!(model)
        c = JuMP.value.(x)
    end

    status           = termination_status(model)
    genetic_gain     = dot(c, g)
    genetic_variance = c' * G * c

    return (
        contributions    = c,
        time_decomp      = time_decomp,
        time_opt         = time_opt,
        time_total       = time_decomp + time_opt,
        status           = status,
        genetic_gain     = genetic_gain,
        genetic_variance = genetic_variance,
        rank             = rank,
        method           = "PCA Randomized"
    )
end

# =============================================================================
# 5. COMPARISON UTILITIES
# =============================================================================

function compare_to_baseline(baseline, approx, G)
    rmse      = sqrt(mean((approx.contributions .- baseline.contributions).^2))
    gain_diff = abs(approx.genetic_gain - baseline.genetic_gain) / abs(baseline.genetic_gain)
    var_diff  = abs(approx.genetic_variance - baseline.genetic_variance) / abs(baseline.genetic_variance)

    thr = 1e-4
    base_sel  = findall(baseline.contributions .> thr)
    approx_sel = findall(approx.contributions .> thr)
    overlap    = length(intersect(base_sel, approx_sel))
    overlap_pct = overlap / max(length(base_sel), 1) * 100

    speedup = baseline.time_total / approx.time_total

    # Rank correlation of contributions (Spearman)
    n = length(baseline.contributions)
    rank_base   = sortperm(sortperm(baseline.contributions))
    rank_approx = sortperm(sortperm(approx.contributions))
    spearman_rho = cor(Float64.(rank_base), Float64.(rank_approx))

    return (
        rmse         = rmse,
        gain_diff_pct = gain_diff * 100,
        var_diff_pct  = var_diff * 100,
        overlap       = overlap,
        overlap_pct   = overlap_pct,
        speedup       = speedup,
        spearman_rho  = spearman_rho
    )
end

# =============================================================================
# 6. MAIN ANALYSIS
# =============================================================================

println("="^70)
println("Running OCS Analysis")
println("="^70)
println()

# --- Baseline: Full Dense ----------------------------------------------------
println("Computing baseline (Full Dense, sex-constrained)...")
baseline = ocs_full_dense_sex(G, g_vec, male_idx, female_idx, N_select;
                               gamma=gamma_default, verbose=true)

@printf("  Status        : %s\n",  string(baseline.status))
@printf("  Genetic gain  : %.6f\n", baseline.genetic_gain)
@printf("  Genetic var   : %.6f\n", baseline.genetic_variance)
@printf("  ΔF per gen    : %.4f%%\n", 1/(2 * N_select^2 / (2*baseline.genetic_variance)) * 100)
@printf("  Time (total)  : %.2f s\n", baseline.time_total)

n_sel_m = sum(baseline.contributions[male_idx]   .> 1e-4)
n_sel_f = sum(baseline.contributions[female_idx] .> 1e-4)
@printf("  Selected males  : %d\n", n_sel_m)
@printf("  Selected females: %d\n", n_sel_f)
println()

# --- Results DataFrame -------------------------------------------------------
results = DataFrame(
    rank          = Int[],
    method        = String[],
    replicate     = Int[],
    time_decomp   = Float64[],
    time_opt      = Float64[],
    time_total    = Float64[],
    genetic_gain  = Float64[],
    genetic_var   = Float64[],
    rmse          = Float64[],
    gain_diff_pct = Float64[],
    var_diff_pct  = Float64[],
    overlap_pct   = Float64[],
    spearman_rho  = Float64[],
    speedup       = Float64[],
    status        = String[]
)

# Add baseline row
push!(results, (
    rank=0, method="Full Dense", replicate=1,
    time_decomp=0.0, time_opt=baseline.time_total, time_total=baseline.time_total,
    genetic_gain=baseline.genetic_gain, genetic_var=baseline.genetic_variance,
    rmse=0.0, gain_diff_pct=0.0, var_diff_pct=0.0,
    overlap_pct=100.0, spearman_rho=1.0, speedup=1.0,
    status=string(baseline.status)
))

# --- Rank comparison ---------------------------------------------------------
println("Running rank comparison...")
println("Ranks: $ranks_to_test | Replicates: $n_replicates")
println()

for r in ranks_to_test
    println("Rank $r:")

    # PCA Standard (deterministic — run once)
    print("  PCA-Std ... ")
    sol_std = ocs_pca_standard_sex(G, g_vec, male_idx, female_idx, N_select;
                                   rank=r, gamma=gamma_default)
    cmp = compare_to_baseline(baseline, sol_std, G)
    push!(results, (
        rank=r, method="PCA Standard", replicate=1,
        time_decomp=sol_std.time_decomp, time_opt=sol_std.time_opt, time_total=sol_std.time_total,
        genetic_gain=sol_std.genetic_gain, genetic_var=sol_std.genetic_variance,
        rmse=cmp.rmse, gain_diff_pct=cmp.gain_diff_pct, var_diff_pct=cmp.var_diff_pct,
        overlap_pct=cmp.overlap_pct, spearman_rho=cmp.spearman_rho,
        speedup=cmp.speedup, status=string(sol_std.status)
    ))
    @printf("gain_diff=%.4f%%  speedup=%.2fx  ρ=%.4f ✓\n",
            cmp.gain_diff_pct, cmp.speedup, cmp.spearman_rho)

    # PCA Randomized — multiple replicates to capture RSVD variance
    for rep in 1:n_replicates
        print("  PCA-Rand  rep $rep/$n_replicates ... ")
        sol_rand = ocs_pca_randomized_sex(G, g_vec, male_idx, female_idx, N_select;
                                          rank=r, gamma=gamma_default)
        cmp = compare_to_baseline(baseline, sol_rand, G)
        push!(results, (
            rank=r, method="PCA Randomized", replicate=rep,
            time_decomp=sol_rand.time_decomp, time_opt=sol_rand.time_opt, time_total=sol_rand.time_total,
            genetic_gain=sol_rand.genetic_gain, genetic_var=sol_rand.genetic_variance,
            rmse=cmp.rmse, gain_diff_pct=cmp.gain_diff_pct, var_diff_pct=cmp.var_diff_pct,
            overlap_pct=cmp.overlap_pct, spearman_rho=cmp.spearman_rho,
            speedup=cmp.speedup, status=string(sol_rand.status)
        ))
        @printf("gain_diff=%.4f%%  speedup=%.2fx  ρ=%.4f ✓\n",
                cmp.gain_diff_pct, cmp.speedup, cmp.spearman_rho)
    end
    println()
end

# =============================================================================
# 7. SUMMARY TABLE
# =============================================================================

println("="^70)
println("SUMMARY — Mean across replicates (PCA Randomized)")
println("="^70)

rsvd_results = filter(r -> r.method == "PCA Randomized", results)
summary_df = combine(
    groupby(rsvd_results, :rank),
    :gain_diff_pct => mean => :mean_gain_diff_pct,
    :gain_diff_pct => std  => :sd_gain_diff_pct,
    :speedup       => mean => :mean_speedup,
    :spearman_rho  => mean => :mean_spearman_rho,
    :overlap_pct   => mean => :mean_overlap_pct
)

println()
println(rpad("Rank", 6), rpad("mean_gain_diff%", 18), rpad("sd", 10),
        rpad("speedup", 10), rpad("Spearman ρ", 12), "overlap%")
println("-"^60)
for row in eachrow(summary_df)
    @printf("%-6d %-18.4f %-10.4f %-10.2f %-12.4f %.1f\n",
            row.rank, row.mean_gain_diff_pct, row.sd_gain_diff_pct,
            row.mean_speedup, row.mean_spearman_rho, row.mean_overlap_pct)
end
println()

# =============================================================================
# 8. BEST RSVD SOLUTION — pick rank with gain_diff < 1% and max speedup
# =============================================================================

good_ranks = filter(r -> r.mean_gain_diff_pct < 1.0, eachrow(summary_df))
if !isempty(good_ranks)
    best_row = sort(collect(good_ranks), by=r -> -r.mean_speedup)[1]
    best_rank = best_row.rank
    println("Recommended rank: $best_rank  (gain diff < 1%, max speedup)")
else
    best_rank = summary_df[end, :rank]
    println("Warning: no rank achieves <1% gain diff. Using max rank = $best_rank")
end

println("Re-running best RSVD at rank $best_rank for contribution export...")
best_sol = ocs_pca_randomized_sex(G, g_vec, male_idx, female_idx, N_select;
                                   rank=best_rank, gamma=gamma_default, verbose=true)

# =============================================================================
# 9. EXPORT
# =============================================================================

# Rank comparison table
csv_rank = joinpath(OUT_DIR, "ocs_rank_comparison_sexconstrained.csv")
CSV.write(csv_rank, results)
println("Saved rank comparison: $csv_rank")

# Full Dense contributions
contrib_full = DataFrame(
    ID           = all_ids,
    sex          = all_sex,
    GEBV         = g_vec,
    contribution = baseline.contributions
)
sort!(contrib_full, :contribution, rev=true)
csv_full = joinpath(OUT_DIR, "ocs_contributions_full_dense.csv")
CSV.write(csv_full, contrib_full)
println("Saved Full Dense contributions: $csv_full")

# Best RSVD contributions
contrib_rsvd = DataFrame(
    ID           = all_ids,
    sex          = all_sex,
    GEBV         = g_vec,
    contribution = best_sol.contributions
)
sort!(contrib_rsvd, :contribution, rev=true)
csv_rsvd = joinpath(OUT_DIR, "ocs_contributions_rsvd_rank$(best_rank).csv")
CSV.write(csv_rsvd, contrib_rsvd)
println("Saved RSVD contributions: $csv_rsvd")

# JLD2 full results
jld_file = joinpath(OUT_DIR, "ocs_qtlmas_$(Dates.format(Dates.now(), "yyyymmdd_HHMMSS")).jld2")
jldsave(jld_file;
    results       = results,
    baseline      = baseline,
    best_sol      = best_sol,
    summary       = summary_df,
    spectrum      = spec_df,
    metadata      = Dict("n" => n_total, "N_select" => N_select, "gamma" => gamma_default,
                         "n_males" => n_males, "n_females" => n_females,
                         "spectrum_rank" => spectrum_rank,
                         "tr_G" => tr_G, "eff_rank" => eff_rank,
                         "timestamp" => Dates.now())
)
println("Saved JLD2: $jld_file")

println()
println("="^70)
println("ANALYSIS COMPLETE")
println("="^70)
