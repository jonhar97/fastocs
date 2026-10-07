# =============================================================================
# QTLMAS 2010 — Gamma × Rank Sweep for Theoretical Speedup Analysis
#
# Research question:
#   Does tightening the diversity constraint (lower gamma) increase the
#   computational advantage of RSVD-OCS by concentrating the OCS solution
#   on fewer individuals, making the GRM better approximated at low rank?
#
# Design:
#   For each gamma in a logarithmic range, run the full-dense baseline and
#   the RSVD-OCS across a range of ranks. Record accuracy (gain_diff_pct),
#   speedup, number of selected individuals, and Spearman ρ of contributions.
#   Identify the recommended rank per gamma (first rank where mean gain_diff < 1%).
#
# Outputs:
#   gamma_rank_sweep_raw.csv     — one row per gamma × rank × replicate
#   gamma_rank_sweep_summary.csv — mean/sd across replicates, one row per gamma × rank
#   gamma_rank_sweep_elbow.csv   — recommended rank and speedup at elbow, one row per gamma
#
# These CSVs are designed for direct import into R for figure production.
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

BASE_DIR = normpath(joinpath(@__DIR__, "..", "data"))
OUT_DIR  = normpath(joinpath(@__DIR__, "..", "output"))

tbv_file  = joinpath(BASE_DIR, "tbv.txt")
gebv_file = joinpath(BASE_DIR, "GEBV_output.txt")
geno_file = joinpath(BASE_DIR, "QTLMAS2010gen/QTLMAS2010gen.txt")

# Sweep parameters
N_select    = 100
SEL_THRESH  = 1e-4
GAIN_THRESH = 1.0      # % gain difference threshold for recommended rank

# Gammas: logarithmic range from tight to loose diversity constraint.
# Skipping < 0.1 (degenerate 1-3 animal solutions) and > 10 (diffuse regime).
gammas = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

# Ranks to test at each gamma
ranks = [5, 10, 15, 20, 25, 30, 40, 50, 75, 100]

# RSVD replicates per gamma × rank
n_replicates = 5

# =============================================================================
# 1. DATA LOADING
# =============================================================================

println("="^70)
println("QTLMAS — Gamma × Rank Sweep")
println("="^70)
println("Timestamp: $(Dates.now())")
println()
println("Gammas : $gammas")
println("Ranks  : $ranks")
println("Reps   : $n_replicates")
println()

println("[1] Loading data...")
tbv_raw    = readdlm(tbv_file, ',', header=false)
all_ids    = Int.(tbv_raw[:, 1])
all_sex    = Int.(tbv_raw[:, 2])
n_total    = length(all_ids)
male_idx   = findall(all_sex .== 1)
female_idx = findall(all_sex .== 0)

gebv_df    = CSV.read(gebv_file, DataFrame; delim='\t')
gebv_ids   = Int.(gebv_df.ID)
gebv_vals  = Float64.(gebv_df.GEBV)
id_to_gebv = Dict(gebv_ids[i] => gebv_vals[i] for i in eachindex(gebv_ids))
g_vec      = [get(id_to_gebv, id, 0.0) for id in all_ids]

@printf("  n=%d  males=%d  females=%d\n", n_total, length(male_idx), length(female_idx))

println("[2] Building GRM (VanRaden 2008)...")
M      = readdlm(geno_file, ',', header=false)
N_mat  = M .- 1
p_freq = sum(M; dims=1) ./ (n_total * 2)
Z      = N_mat .- 2 .* (p_freq .- 0.5)
denom  = 2 * sum(p_freq .* (1 .- p_freq))
G      = (Z * Z') ./ denom
G      = (G + G') / 2
@printf("  GRM: %d × %d  (diag mean=%.4f)\n\n", n_total, n_total, mean(diag(G)))

# =============================================================================
# 2. SOLVER FUNCTIONS
# =============================================================================

function rsvd(A::Matrix{T}, k::Int; p::Int=10, q::Int=2) where T
    n, m = size(A)
    l    = min(k + p, min(n, m))
    Ω    = randn(T, m, l)
    Y    = A * Ω
    for _ in 1:q
        Y = A * (A' * Y)
    end
    Q, _ = qr(Y)
    Q    = Matrix(Q)
    B    = Q' * A
    Ũ, S, V = svd(B)
    U    = Q * Ũ
    return U[:, 1:k], S[1:k], V[:, 1:k]
end

"""Full dense sex-constrained OCS. Returns named tuple."""
function solve_full_dense(G, g, male_idx, female_idx, N_select; gamma)
    n   = length(g)
    N_m = N_select ÷ 2
    N_f = N_select - N_m
    t   = @elapsed begin
        model = Model(optimizer_with_attributes(OSQP.Optimizer,
            "max_iter" => 20000, "eps_abs" => 1e-6,
            "eps_rel" => 1e-6, "verbose" => false))
        @variable(model, x[1:n] >= 0)
        @objective(model, Min,
            0.5 * (x' * G * x) - (1/(2*gamma)) * dot(g, x))
        @constraint(model, sum(x[i] for i in male_idx)   == Float64(N_m))
        @constraint(model, sum(x[i] for i in female_idx) == Float64(N_f))
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

"""RSVD sex-constrained OCS with auxiliary variable formulation."""
function solve_rsvd(G, g, male_idx, female_idx, N_select; rank, gamma)
    n   = length(g)
    N_m = N_select ÷ 2
    N_f = N_select - N_m

    t_decomp = @elapsed begin
        U, S, _ = rsvd(G, rank; p=10, q=2)
        F       = U * Diagonal(sqrt.(S))          # n × rank
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
        @constraint(model, sum(x[i] for i in male_idx)   == Float64(N_m))
        @constraint(model, sum(x[i] for i in female_idx) == Float64(N_f))
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

    # Spearman ρ on full contribution vector
    rb = sortperm(sortperm(baseline.contributions))
    ra = sortperm(sortperm(approx.contributions))
    spearman_rho  = cor(Float64.(rb), Float64.(ra))

    # Overlap among contributors above threshold
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

println("="^70)
println("Running gamma × rank sweep")
println("="^70)
println()

# Raw results: one row per gamma × rank × replicate
raw_rows = []

for gamma in gammas

    println("━"^70)
    @printf("  gamma = %.2g\n", gamma)
    println("━"^70)

    # --- Baseline ---
    print("  Full Dense baseline ... ")
    baseline = solve_full_dense(G, g_vec, male_idx, female_idx, N_select; gamma=gamma)

    if baseline.status != "OPTIMAL"
        @printf("FAILED (%s) — skipping gamma\n\n", baseline.status)
        continue
    end

    n_sel_m = sum(baseline.contributions[male_idx]   .> SEL_THRESH)
    n_sel_f = sum(baseline.contributions[female_idx] .> SEL_THRESH)
    @printf("done (%.1fs)  gain=%.2f  var=%.4f  sel=%d+%d\n",
            baseline.time_total, baseline.genetic_gain, baseline.genetic_var,
            n_sel_m, n_sel_f)

    # --- Rank sweep ---
    for r in ranks
        print("  rank $r: ")
        for rep in 1:n_replicates
            sol  = solve_rsvd(G, g_vec, male_idx, female_idx, N_select;
                              rank=r, gamma=gamma)
            cmp  = compare(baseline, sol, G)
            n_sel_approx = sum(sol.contributions .> SEL_THRESH)

            push!(raw_rows, (
                gamma         = gamma,
                rank          = r,
                replicate     = rep,
                n_sel_baseline = n_sel_m + n_sel_f,
                n_sel_approx  = n_sel_approx,
                gain_diff_pct = cmp.gain_diff_pct,
                var_diff_pct  = cmp.var_diff_pct,
                speedup       = cmp.speedup,
                spearman_rho  = cmp.spearman_rho,
                overlap_pct   = cmp.overlap_pct,
                time_baseline = baseline.time_total,
                time_rsvd     = sol.time_total,
                time_decomp   = sol.time_decomp,
                time_opt      = sol.time_opt,
                status        = sol.status
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
# 5. ELBOW TABLE — recommended rank per gamma
# =============================================================================

# For each gamma, find the lowest rank where mean gain_diff < GAIN_THRESH%,
# then record speedup and n_selected at that rank.

println("="^70)
println("Elbow table — recommended rank per gamma (gain_diff < $(GAIN_THRESH)%)")
println("="^70)
println()
println(rpad("gamma", 8), rpad("n_sel", 8), rpad("rec_rank", 10),
        rpad("speedup", 10), rpad("gain_diff%", 12), rpad("spearman_ρ", 12), "overlap%")
println("-"^68)

elbow_rows = []

for gamma in gammas
    sub = filter(r -> r.gamma == gamma, summary_df)
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
    @printf("%s%-7.2g %-8d %-10d %-10.2f %-12.4f %-12.4f %.1f\n",
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

raw_csv     = joinpath(OUT_DIR, "gamma_rank_sweep_raw.csv")
summary_csv = joinpath(OUT_DIR, "gamma_rank_sweep_summary.csv")
elbow_csv   = joinpath(OUT_DIR, "gamma_rank_sweep_elbow.csv")

CSV.write(raw_csv,     raw_df)
CSV.write(summary_csv, summary_df)
CSV.write(elbow_csv,   elbow_df)

@printf("Saved raw results    : %s\n", raw_csv)
@printf("Saved summary table  : %s\n", summary_csv)
@printf("Saved elbow table    : %s\n", elbow_csv)

println()
println("="^70)
println("SWEEP COMPLETE  —  $(Dates.now())")
println("="^70)
println()
println("Suggested R figures from these CSVs:")
println("  1. gain_diff_pct vs rank, coloured by gamma  (gamma_rank_sweep_summary.csv)")
println("     → shows convergence curves shifting left as gamma decreases")
println("  2. speedup at recommended rank vs gamma      (gamma_rank_sweep_elbow.csv)")
println("     → shows the inverse relationship between diversity constraint and speedup")
println("  3. n_sel_baseline vs recommended rank        (gamma_rank_sweep_elbow.csv)")
println("     → connects solution sparsity to rank requirement")
