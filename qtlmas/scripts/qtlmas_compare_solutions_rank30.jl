# =============================================================================
# QTLMAS 2010 — Solution Comparison at Recommended Rank (gamma = 1, rank = 30)
#
# Adapted from compare_solutions_rank30.jl (Norway spruce analysis).
# Loads pre-computed results from the JLD2 saved by qtlmas_ocs_sexconstrained.jl
# and produces publication-ready figures using CairoMakie.
#
# Key differences from Norway spruce version:
#   - Sex-constrained OCS (separate male/female contribution vectors)
#   - Single trait GEBV (not a multi-trait selection index)
#   - Loads from JLD2 rather than requiring functions in scope
#   - CairoMakie instead of Plots.jl
#
# Figures produced:
#   Fig A: Contribution scatter — Full Dense vs RSVD (coloured by sex)
#   Fig B: Ranking shift scatter — top contributors in Full Dense vs their
#          rank under RSVD (coloured by sex)
#   Fig C: Cumulative contribution curves — Full Dense vs PCA Std vs RSVD
#
# OUTPUT:
#   qtlmas_solution_comparison_rank30.pdf
#   qtlmas_solution_comparison_rank30.csv  (summary statistics)
# =============================================================================

using CairoMakie
using JLD2
using CSV
using DataFrames
using Statistics
using StatsBase
using Printf

# =============================================================================
# 0. CONFIGURATION
# =============================================================================

BASE_DIR = normpath(joinpath(@__DIR__, "..", "data"))
OUT_DIR  = normpath(joinpath(@__DIR__, "..", "output"))

# JLD2 file saved by qtlmas_ocs_sexconstrained.jl (gamma=1, includes rank sweep)
# Update filename to match your actual saved file
jld_file = joinpath(OUT_DIR, "ocs_qtlmas_20260414_062958.jld2")

# OCS settings — must match what was used in the main script
gamma_val    = 1.0
rec_rank     = 30       # recommended rank from the sweep
SEL_THRESH   = 1e-2
TOP_N        = 50       # total selected in full dense at gamma=1 (41M + 37F)

# Okabe-Ito colours (colour-blind safe)
COL_MALE   = RGBf(0.000, 0.447, 0.698)   # blue
COL_FEMALE = RGBf(0.835, 0.369, 0.000)   # vermillion
COL_FULL   = RGBf(0.000, 0.000, 0.000)   # black
COL_STD    = RGBf(0.000, 0.620, 0.451)   # green
COL_RAND   = RGBf(0.800, 0.475, 0.655)   # mauve

# =============================================================================
# 1. LOAD DATA
# =============================================================================

println("="^70)
println("QTLMAS — Solution Comparison at Rank $rec_rank (gamma = $gamma_val)")
println("="^70)
println()

println("[1] Loading JLD2 results...")
data = load(jld_file)

baseline = data["baseline"]          # full dense solution
best_sol = data["best_sol"]          # RSVD at recommended rank
results  = data["results"]           # full rank comparison DataFrame
meta     = data["metadata"]

n_total    = meta["n"]
n_males    = meta["n_males"]
n_females  = meta["n_females"]

@printf("  n=%d  males=%d  females=%d  gamma=%.1f\n",
        n_total, n_males, n_females, meta["gamma"])
@printf("  Baseline gain=%.4f  var=%.6f\n",
        baseline.genetic_gain, baseline.genetic_variance)
println()

# Reconstruct sex index vectors from metadata dimensions
# Assumes tbv.txt order: all_sex column determines male_idx / female_idx
# We rebuild from scratch to be safe
tbv_file   = joinpath(BASE_DIR, "tbv.txt")
using DelimitedFiles
tbv_raw    = readdlm(tbv_file, ',', header=false)
all_ids    = Int.(tbv_raw[:, 1])
all_sex    = Int.(tbv_raw[:, 2])
male_idx   = findall(all_sex .== 1)
female_idx = findall(all_sex .== 0)

# GEBV vector
gebv_file  = joinpath(BASE_DIR, "GEBV_output.txt")
gebv_df    = CSV.read(gebv_file, DataFrame; delim='\t')
id_to_gebv = Dict(Int(gebv_df.ID[i]) => Float64(gebv_df.GEBV[i]) for i in eachindex(gebv_df.ID))
g_vec      = [get(id_to_gebv, id, 0.0) for id in all_ids]

# Extract contribution vectors
c_full = baseline.contributions
c_rand = best_sol.contributions

# Re-run PCA Standard at rec_rank for comparison
# (best_sol is RSVD; we need std for the 3-way comparison)
# Load from results DataFrame instead — pick PCA Standard at rec_rank
std_row = filter(r -> r.method == "PCA Standard" && r.rank == rec_rank, results)
if isempty(std_row)
    @warn "PCA Standard at rank $rec_rank not found in results — skipping std comparisons"
    has_std = false
    c_std   = c_rand   # fallback
else
    has_std = true
    # Contributions not stored in the summary DataFrame — need to re-run
    # Load the saved CSV contributions file for full dense; re-run std here
    println("[2] PCA Standard contributions not in JLD2 summary — re-running at rank $rec_rank...")

    # Rebuild G (required for solver)
    using LinearAlgebra
    geno_file = joinpath(BASE_DIR, "QTLMAS2010gen", "QTLMAS2010gen.txt")
    println("    Building GRM...")
    M      = readdlm(geno_file, ',', header=false)
    N_mat  = M .- 1
    p_freq = sum(M; dims=1) ./ (n_total * 2)
    Z      = N_mat .- 2 .* (p_freq .- 0.5)
    denom  = 2 * sum(p_freq .* (1 .- p_freq))
    G      = (Z * Z') ./ denom
    G      = (G + G') / 2

    using OSQP, JuMP
    println("    Running PCA Standard OCS at rank $rec_rank...")
    N_select = meta["N_select"]
    N_m      = N_select ÷ 2
    N_f      = N_select - N_m

    t_std = @elapsed begin
        F_svd  = svd(G)
        U_std  = F_svd.U[:, 1:rec_rank]
        S_std  = F_svd.S[1:rec_rank]
        F_std  = U_std * Diagonal(sqrt.(S_std))
        D_std  = diag(G) - vec(sum(F_std.^2, dims=2))

        model = Model(optimizer_with_attributes(OSQP.Optimizer,
            "max_iter" => 20000, "eps_abs" => 1e-6,
            "eps_rel"  => 1e-6, "verbose"  => false))
        @variable(model, x[1:n_total] >= 0)
        @variable(model, y[1:rec_rank])
        @objective(model, Min,
            0.5 * sum(D_std[i] * x[i]^2 for i in 1:n_total) +
            0.5 * dot(y, y) - (1/(2*gamma_val)) * dot(g_vec, x))
        @constraint(model, sum(x[i] for i in male_idx)   == Float64(N_m))
        @constraint(model, sum(x[i] for i in female_idx) == Float64(N_f))
        for j in 1:rec_rank
            @constraint(model, y[j] == sum(F_std[i,j] * x[i] for i in 1:n_total))
        end
        JuMP.optimize!(model)
        c_std = JuMP.value.(x)
    end
    @printf("    Done (%.1f s)\n", t_std)
end
println()

# =============================================================================
# 2. SUMMARY STATISTICS
# =============================================================================

println("="^70)
println("Summary statistics")
println("="^70)

function sel_count(c, idx=1:length(c))
    sum(c[idx] .> SEL_THRESH)
end

n_sel_full_m = sel_count(c_full, male_idx)
n_sel_full_f = sel_count(c_full, female_idx)
n_sel_rand_m = sel_count(c_rand, male_idx)
n_sel_rand_f = sel_count(c_rand, female_idx)
n_sel_std_m  = has_std ? sel_count(c_std, male_idx)   : missing
n_sel_std_f  = has_std ? sel_count(c_std, female_idx) : missing

selected_full = findall(c_full .> SEL_THRESH)
selected_rand = findall(c_rand .> SEL_THRESH)
selected_std  = has_std ? findall(c_std  .> SEL_THRESH) : selected_rand

overlap_rand = length(intersect(selected_rand, selected_full)) /
               length(selected_full) * 100
overlap_std  = has_std ?
               length(intersect(selected_std, selected_full)) /
               length(selected_full) * 100 : NaN

rho_rand = corspearman(c_rand, c_full)
rho_std  = has_std ? corspearman(c_std, c_full) : NaN

gain_diff_rand = abs(best_sol.genetic_gain - baseline.genetic_gain) /
                 abs(baseline.genetic_gain) * 100

@printf("\n  %-28s %12s %12s %12s\n", "Metric", "Full Dense", "PCA Std", "PCA Rand")
println("  " * "-"^66)
@printf("  %-28s %12.4f %12s %12.4f\n", "Genetic gain",
        baseline.genetic_gain, has_std ? @sprintf("%.4f", dot(c_std, g_vec)) : "—",
        best_sol.genetic_gain)
@printf("  %-28s %12.6f %12s %12.6f\n", "Genetic variance",
        baseline.genetic_variance,
        has_std ? @sprintf("%.6f", c_std' * G * c_std) : "—",
        best_sol.genetic_variance)
@printf("  %-28s %12d %12s %12d\n", "Selected males",
        n_sel_full_m, has_std ? string(n_sel_std_m) : "—", n_sel_rand_m)
@printf("  %-28s %12d %12s %12d\n", "Selected females",
        n_sel_full_f, has_std ? string(n_sel_std_f) : "—", n_sel_rand_f)
@printf("  %-28s %12s %12.1f %12.1f\n", "Overlap with full (%)",
        "100.0", overlap_std, overlap_rand)
@printf("  %-28s %12s %12.4f %12.4f\n", "Spearman ρ (vs full)",
        "1.0000", rho_std, rho_rand)
@printf("  %-28s %12s %12s %12.4f\n", "Gain diff (%)",
        "—", "—", gain_diff_rand)
println()

# =============================================================================
# 3. RANKING ANALYSIS — top contributors
# =============================================================================

# Rank all individuals by contribution under full dense
order_full = sortperm(c_full, rev=true)
order_rand = sortperm(c_rand, rev=true)

# For the top TOP_N individuals in full dense, find their rank under RSVD
rank_in_rand = zeros(Int, TOP_N)
for i in 1:TOP_N
    ind = order_full[i]
    rank_in_rand[i] = findfirst(order_rand .== ind)
end
rank_shifts = rank_in_rand .- (1:TOP_N)

@printf("  Rank shifts for top %d contributors (full dense → RSVD rank %d):\n",
        TOP_N, rec_rank)
@printf("    Mean |shift| : %.1f\n",  mean(abs.(rank_shifts)))
@printf("    Median shift : %.1f\n",  median(rank_shifts))
@printf("    Max |shift|  : %d\n",    maximum(abs.(rank_shifts)))
@printf("    Spearman ρ   : %.4f\n",  rho_rand)
println()

# =============================================================================
# 4. FIGURES
# =============================================================================

println("[3] Producing figures...")

fig = Figure(size = (1050, 340), fontsize = 11)

ax_A = Axis(fig[1, 1],
    xlabel = "Full dense contribution",
    ylabel = "RSVD contribution  (rank $rec_rank)",
    title  = "A  Contribution concordance")

ax_B = Axis(fig[1, 2],
    xlabel = "Rank in full dense solution",
    ylabel = "Rank shift  (RSVD − full dense)",
    title  = "B  Ranking shifts — top $TOP_N contributors")

ax_C = Axis(fig[1, 3],
    xlabel = "Number of individuals (sorted by contribution)",
    ylabel = "Cumulative contribution (%)",
    title  = "C  Cumulative contribution distribution")

# --- Panel A: contribution scatter coloured by sex ---
# Plot all individuals selected by at least one method (union), keeping
# contributions paired so each point represents the same individual in both.
# This means the scatter directly reflects the Spearman ρ reported above.
mask_m = (c_full[male_idx]   .> SEL_THRESH) .| (c_rand[male_idx]   .> SEL_THRESH)
mask_f = (c_full[female_idx] .> SEL_THRESH) .| (c_rand[female_idx] .> SEL_THRESH)

c_full_m = c_full[male_idx[mask_m]]
c_rand_m = c_rand[male_idx[mask_m]]
c_full_f = c_full[female_idx[mask_f]]
c_rand_f = c_rand[female_idx[mask_f]]

@printf("  Panel A: %d males, %d females plotted (selected by at least one method)\n",
        sum(mask_m), sum(mask_f))

scatter!(ax_A, c_full_m, c_rand_m;
         color = COL_MALE, markersize = 7, alpha = 0.8, label = "Male")
scatter!(ax_A, c_full_f, c_rand_f;
         color = COL_FEMALE, markersize = 7, alpha = 0.8, label = "Female")

# Identity line
cmax = max(maximum(vcat(c_full_m, c_full_f)), maximum(vcat(c_rand_m, c_rand_f))) * 1.08
lines!(ax_A, [0, cmax], [0, cmax];
       color = (:black, 0.5), linestyle = :dash, linewidth = 1.5)

# Annotation
text!(ax_A, @sprintf("ρ = %.3f", rho_rand);
      position = (cmax * 0.05, cmax * 0.90), fontsize = 10)

axislegend(ax_A; position = :rb, framevisible = false, labelsize = 9)

# --- Panel B: ranking shifts coloured by sex ---
# Classify each of the top TOP_N by sex
top_ids   = order_full[1:TOP_N]
top_sex   = all_sex[top_ids]   # 1=male, 0=female

male_mask_top   = top_sex .== 1
female_mask_top = top_sex .== 0

scatter!(ax_B, (1:TOP_N)[male_mask_top],   rank_shifts[male_mask_top];
         color = COL_MALE,   markersize = 6, alpha = 0.7, label = "Male")
scatter!(ax_B, (1:TOP_N)[female_mask_top], rank_shifts[female_mask_top];
         color = COL_FEMALE, markersize = 6, alpha = 0.7, label = "Female")

hlines!(ax_B, [0]; color = (:black, 0.4), linestyle = :dash, linewidth = 1.5)

text!(ax_B, @sprintf("Mean |shift| = %.1f\nMax |shift| = %d",
      mean(abs.(rank_shifts)), maximum(abs.(rank_shifts)));
      position = (TOP_N * 0.55, maximum(rank_shifts) * 0.85), fontsize = 9)

axislegend(ax_B; position = :rt, framevisible = false, labelsize = 9)

# --- Panel C: cumulative contributions ---
function cumcontrib(c, idx)
    vals = sort(c[idx[c[idx] .> SEL_THRESH]], rev=true)
    isempty(vals) && return [0.0], [0.0]
    return 1:length(vals), cumsum(vals) ./ sum(vals) .* 100
end

x_f, y_f = cumcontrib(c_full, 1:n_total)
x_s, y_s = has_std ? cumcontrib(c_std,  1:n_total) : (x_f, y_f)
x_r, y_r = cumcontrib(c_rand, 1:n_total)

lines!(ax_C, x_f, y_f; color = COL_FULL,  linewidth = 2.0, label = "Full dense")
has_std && lines!(ax_C, x_s, y_s;
                  color = COL_STD,   linewidth = 2.0, linestyle = :dash,
                  label = "PCA Standard")
lines!(ax_C, x_r, y_r; color = COL_RAND,  linewidth = 2.0, linestyle = :dot,
       label = "RSVD  (rank $rec_rank)")
hlines!(ax_C, [50.0]; color = (:black, 0.3), linestyle = :dashdot,
        linewidth = 1.0)

axislegend(ax_C; position = :rb, framevisible = false, labelsize = 9)

colgap!(fig.layout, 14)

fig_path = joinpath(OUT_DIR, "qtlmas_solution_comparison_rank$(rec_rank).pdf")
save(fig_path, fig; pt_per_unit = 1)
println("  Saved: $fig_path")

# =============================================================================
# 5. EXPORT SUMMARY CSV
# =============================================================================

summary_df = DataFrame(
    metric    = ["Genetic gain", "Genetic variance",
                 "Selected males", "Selected females",
                 "Overlap with full (%)", "Spearman rho", "Gain diff (%)"],
    full_dense = [baseline.genetic_gain, baseline.genetic_variance,
                  n_sel_full_m, n_sel_full_f, 100.0, 1.0, 0.0],
    pca_rand   = [best_sol.genetic_gain, best_sol.genetic_variance,
                  n_sel_rand_m, n_sel_rand_f, overlap_rand, rho_rand,
                  gain_diff_rand]
)

csv_path = joinpath(OUT_DIR, "qtlmas_solution_comparison_rank$(rec_rank).csv")
CSV.write(csv_path, summary_df)
println("  Saved: $csv_path")

println()
println("="^70)
println("DONE")
println("="^70)

display(fig)
