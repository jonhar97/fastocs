# =============================================================================
# QTLMAS — Gamma × Rank Sweep Visualization
#
# Produces a three-panel figure from the gamma_rank_sweep CSVs.
#
# Panel A: Gain difference (%) vs rank, one curve per gamma (log-y scale)
#          Horizontal dashed line at 1% threshold.
#          → Shows convergence curves shifting left as gamma decreases.
#
# Panel B: Speedup vs rank, one curve per gamma
#          → Shows that tight diversity constraints give high speedup
#            even at very low ranks.
#
# Panel C: Speedup at recommended rank vs number of selected individuals
#          One labelled point per gamma.
#          → Connects solution sparsity to computational advantage.
#
# Requires: CairoMakie, CSV, DataFrames
# =============================================================================

using CairoMakie
using CSV
using DataFrames
using Statistics
using Printf

# =============================================================================
# 0. PATHS
# =============================================================================

OUT_DIR     = "C:/Users/JOAH/OneDrive - Skogforsk/Documents/Projekt/Optimum contribution selection/LowRankApproxOCS/results/QTLMAS/"
summary_csv = joinpath(OUT_DIR, "gamma_rank_sweep_summary.csv")
elbow_csv   = joinpath(OUT_DIR, "gamma_rank_sweep_elbow.csv")
fig_out     = joinpath(OUT_DIR, "gamma_rank_sweep_figure.pdf")

GAIN_THRESH = 1.0   # % — dashed reference line in panel A

# =============================================================================
# 1. LOAD DATA
# =============================================================================

summary = CSV.read(summary_csv, DataFrame)
elbow   = CSV.read(elbow_csv,   DataFrame)

gammas  = sort(unique(summary.gamma))
n_gamma = length(gammas)

# Colour palette: perceptually uniform, colour-blind friendly (Okabe-Ito)
palette = [
    RGBf(0.000, 0.447, 0.698),   # blue       γ=0.1
    RGBf(0.835, 0.369, 0.000),   # vermillion γ=0.5
    RGBf(0.000, 0.620, 0.451),   # green      γ=1.0
    RGBf(0.800, 0.475, 0.655),   # mauve      γ=2.0
    RGBf(0.941, 0.894, 0.259),   # yellow     γ=5.0
    RGBf(0.337, 0.706, 0.914),   # sky blue   γ=10.0
]
gamma_color = Dict(gammas[i] => palette[i] for i in 1:n_gamma)
gamma_label = Dict(g => (g == floor(g) ? @sprintf("γ = %.0f", g) :
                                          @sprintf("γ = %.1f", g))
                   for g in gammas)

# =============================================================================
# 2. FIGURE LAYOUT
# =============================================================================

fig = Figure(size = (1000, 320), fontsize = 11)

ax_A = Axis(fig[1, 1],
    xlabel       = "Rank (k)",
    ylabel       = "Gain difference (%)",
    title        = "A  Approximation accuracy vs rank",
    yscale       = log10,
    ytickformat  = values -> [@sprintf("%.2g", v) for v in values],
    xticks       = [5, 10, 20, 30, 50, 75, 100],
    xminorgridvisible = true,
    yminorgridvisible = true
)

ax_B = Axis(fig[1, 2],
    xlabel       = "Rank (k)",
    ylabel       = "Speedup vs full dense (×)",
    title        = "B  Computational speedup vs rank",
    xticks       = [5, 10, 20, 30, 50, 75, 100],
    xminorgridvisible = true,
    yminorgridvisible = true
)

ax_C = Axis(fig[1, 3],
    xlabel       = "Number of selected individuals",
    ylabel       = "Speedup at recommended rank (×)",
    title        = "C  Speedup vs solution sparsity",
    xminorgridvisible = true,
    yminorgridvisible = true
)

# =============================================================================
# 3. PANEL A — gain_diff vs rank, coloured by gamma
# =============================================================================

hlines!(ax_A, [GAIN_THRESH];
        color = (:black, 0.4), linestyle = :dash, linewidth = 1.2,
        label = "1% threshold")

for gamma in gammas
    sub  = sort(filter(r -> r.gamma == gamma, summary), :rank)
    col  = gamma_color[gamma]
    lab  = gamma_label[gamma]

    # Ribbon: mean ± sd (clipped to log-safe minimum)
    lo = max.(sub.mean_gain_diff_pct .- sub.sd_gain_diff_pct, 1e-4)
    hi = sub.mean_gain_diff_pct .+ sub.sd_gain_diff_pct

    band!(ax_A, sub.rank, lo, hi; color = (col, 0.15))
    lines!(ax_A, sub.rank, sub.mean_gain_diff_pct;
           color = col, linewidth = 2, label = lab)
    scatter!(ax_A, sub.rank, sub.mean_gain_diff_pct;
             color = col, markersize = 6)
end

# Mark recommended rank per gamma (open circle on the curve)
for row in eachrow(elbow)
    row.below_threshold || continue   # skip gammas that never reach threshold
    col = gamma_color[row.gamma]
    sub = filter(r -> r.gamma == row.gamma && r.rank == row.rec_rank, summary)
    isempty(sub) && continue
    scatter!(ax_A, [row.rec_rank], [sub[1, :mean_gain_diff_pct]];
             color = :white, strokecolor = col, strokewidth = 2,
             markersize = 10, marker = :circle)
end

ylims!(ax_A, (1e-2, 200))

# =============================================================================
# 4. PANEL B — speedup vs rank, coloured by gamma
# =============================================================================

for gamma in gammas
    sub = sort(filter(r -> r.gamma == gamma, summary), :rank)
    col = gamma_color[gamma]
    lab = gamma_label[gamma]

    band!(ax_B, sub.rank,
          sub.mean_speedup .- sub.sd_speedup,
          sub.mean_speedup .+ sub.sd_speedup;
          color = (col, 0.15))
    lines!(ax_B, sub.rank, sub.mean_speedup;
           color = col, linewidth = 2, label = lab)
    scatter!(ax_B, sub.rank, sub.mean_speedup;
             color = col, markersize = 6)
end

hlines!(ax_B, [1.0];
        color = (:black, 0.3), linestyle = :dot, linewidth = 1.0)

# =============================================================================
# 5. PANEL C — speedup at recommended rank vs n_selected
# =============================================================================

# Use only gammas that actually reached the threshold
elbow_good = filter(r -> r.below_threshold, elbow)
elbow_all  = elbow   # for reference points that didn't reach threshold

# Fit a simple trend line through the good points for visual guidance
if nrow(elbow_good) >= 2
    x_fit = Float64.(elbow_good.n_sel_baseline)
    y_fit = Float64.(elbow_good.mean_speedup)
    # Log-linear fit: log(speedup) ~ a + b*log(n_sel)
    lx = log.(x_fit)
    b  = (length(lx) * sum(lx .* log.(y_fit)) - sum(lx)*sum(log.(y_fit))) /
         (length(lx) * sum(lx.^2) - sum(lx)^2)
    a  = (sum(log.(y_fit)) - b*sum(lx)) / length(lx)
    x_range = range(minimum(x_fit)*0.8, maximum(x_fit)*1.2, length=100)
    lines!(ax_C, x_range, exp.(a .+ b .* log.(x_range));
           color = (:black, 0.25), linestyle = :dash, linewidth = 1.2)
end

# Points that reached threshold (filled)
for row in eachrow(elbow_good)
    col = gamma_color[row.gamma]
    scatter!(ax_C, [row.n_sel_baseline], [row.mean_speedup];
             color = col, markersize = 12, marker = :circle,
             strokecolor = col, strokewidth = 1)
    text!(ax_C, gamma_label[row.gamma];
          position = (row.n_sel_baseline, row.mean_speedup),
          offset = (6, 3), fontsize = 9, color = col)
end

# Points that did NOT reach threshold (open, greyed)
elbow_bad = filter(r -> !r.below_threshold, elbow)
for row in eachrow(elbow_bad)
    col = gamma_color[row.gamma]
    scatter!(ax_C, [row.n_sel_baseline], [row.mean_speedup];
             color = :white, markersize = 12, marker = :circle,
             strokecolor = col, strokewidth = 2)
    text!(ax_C, gamma_label[row.gamma] * "*";
          position = (row.n_sel_baseline, row.mean_speedup),
          offset = (6, 3), fontsize = 9, color = (col, 0.6))
end

# =============================================================================
# 6. SHARED LEGEND
# =============================================================================

Legend(fig[1, 4],
    [LineElement(color = gamma_color[g], linewidth = 2) for g in gammas],
    [gamma_label[g] for g in gammas],
    "Diversity\nconstraint";
    framevisible = true,
    padding = (8, 8, 8, 8),
    rowgap = 4
)

# =============================================================================
# 7. ANNOTATION AND SAVE
# =============================================================================

# Footnote
Label(fig[2, 1:3],
    "Open circles in A = recommended rank (first rank with gain diff < 1%).  " *
    "Open points in C = γ values where no rank achieved < 1% (shown at rank 100).  " *
    "Ribbons = ±1 SD across $(first(summary.mean_gain_diff_pct |> x -> 5)) replicates.",
    fontsize = 8, color = (:black, 0.5), halign = :left)

rowsize!(fig.layout, 2, Auto(0.08))
colgap!(fig.layout, 10)

save(fig_out, fig; pt_per_unit = 1)
println("Figure saved: $fig_out")
display(fig)
