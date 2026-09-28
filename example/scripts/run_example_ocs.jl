# ==============================================================================
# QUICKSTART: Full-Dense vs RSVD-OCS on the 20-individual synthetic example
#
# Self-contained -- does not `include()` any spruce/scripts file, so it never
# touches the real (large, ungitted) spruce/data files. Mirrors the exact
# OCS / RSVD-OCS formulation used throughout the manuscript (see
# spruce/scripts/spruce_single_gamma_comparison_hblup.jl), just at toy scale
# (n=20) so it runs in well under a second and every number can be inspected
# by eye.
#
# example/data/ already ships with a generated dataset (20 individuals, 100
# markers; see generate_example_data.jl for how it was built), so this
# script just loads it and runs OCS -- nothing to set up first.
#
# Usage:   julia run_example_ocs.jl
# ==============================================================================

using LinearAlgebra, Random, Statistics, Printf, CSV, DataFrames, JuMP, OSQP

# ------------------------------------------------------------------------------
# 1. Load the toy H-matrix and pre-computed EBVs
# ------------------------------------------------------------------------------
data_dir  = joinpath(@__DIR__, "..", "data")
hmat_file = joinpath(data_dir, "Hmat_example_20ind.txt")
ebv_file  = joinpath(data_dir, "EBV_example_20ind.csv")

lines = readlines(hmat_file)
n = length(lines)
ids = Vector{Int}(undef, n)
G = Matrix{Float64}(undef, n, n)
for (i, ln) in enumerate(lines)
    f = split(ln, ',')
    ids[i] = parse(Int, f[1])
    for j in 1:n
        G[i, j] = parse(Float64, f[j + 1])
    end
end

ebv = CSV.read(ebv_file, DataFrame; types = Dict(:ID => Int))
id_to_g = Dict(ebv.ID[i] => ebv.EBV[i] for i in 1:nrow(ebv))
g_vec = [id_to_g[id] for id in ids]

println("Loaded $n individuals, $(length(g_vec)) EBVs from example/data/.")

# ------------------------------------------------------------------------------
# 2. RSVD + OCS QP -- identical formulation to
#    spruce/scripts/spruce_single_gamma_comparison_hblup.jl
# ------------------------------------------------------------------------------
function randomized_svd(A::Matrix{T}, k::Int; p::Int = 5, q::Int = 2) where {T}
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
    model = Model(optimizer_with_attributes(OSQP.Optimizer,
        "max_iter" => 20000, "eps_abs" => 1e-8, "eps_rel" => 1e-8, "verbose" => false))
    @variable(model, x[1:n] >= 0)
    @objective(model, Min, 0.5 * (x' * G * x) - (1 / (2 * gamma)) * dot(g, x))
    @constraint(model, sum(x) == 1.0)
    JuMP.optimize!(model)
    c = JuMP.value.(x)
    return (contributions = c, genetic_gain = dot(c, g), genetic_var = c' * G * c)
end

function solve_from_factor(G, g, F; gamma)
    n, rank = size(F)
    D_vals = diag(G) - vec(sum(F .^ 2, dims = 2))
    model = Model(optimizer_with_attributes(OSQP.Optimizer,
        "max_iter" => 20000, "eps_abs" => 1e-8, "eps_rel" => 1e-8, "verbose" => false))
    @variable(model, x[1:n] >= 0)
    @variable(model, y[1:rank])
    @objective(model, Min,
        0.5 * sum(D_vals[i] * x[i]^2 for i in 1:n) + 0.5 * dot(y, y) - (1 / (2 * gamma)) * dot(g, x))
    @constraint(model, sum(x) == 1.0)
    for j in 1:rank
        @constraint(model, y[j] == sum(F[i, j] * x[i] for i in 1:n))
    end
    JuMP.optimize!(model)
    c = JuMP.value.(x)
    return (contributions = c, genetic_gain = dot(c, g), genetic_var = c' * G * c)
end

# ------------------------------------------------------------------------------
# 3. Run: Full Dense baseline vs RSVD-OCS at a few ranks
# ------------------------------------------------------------------------------
GAMMA      = 8.0     # selects ~45% of the 20 toy candidates -- see header
SEL_THRESH = 1e-4
ranks      = [3, 5, 8, 10, 15]

println("\nFull-dense OCS (gamma=$GAMMA)...")
baseline = solve_full_dense(G, g_vec; gamma = GAMMA)
n_sel_base = sum(baseline.contributions .> SEL_THRESH)
@printf("  n_selected=%d/%d   gain=%.4f   genetic_var=%.4f\n\n",
        n_sel_base, n, baseline.genetic_gain, baseline.genetic_var)

println("Rank sweep: RSVD-OCS vs full dense")
@printf("%6s %10s %12s %10s %8s\n", "rank", "gain", "gain_diff%", "corr(c)", "n_sel")
Random.seed!(123)
for r in ranks
    U, S, _ = randomized_svd(G, r; p = 5, q = 2)
    F = U * Diagonal(sqrt.(S))
    sol = solve_from_factor(G, g_vec, F; gamma = GAMMA)
    gain_diff = abs(sol.genetic_gain - baseline.genetic_gain) / abs(baseline.genetic_gain) * 100
    corr = cor(baseline.contributions, sol.contributions)
    n_sel = sum(sol.contributions .> SEL_THRESH)
    @printf("%6d %10.4f %11.2f%% %10.4f %8d\n", r, sol.genetic_gain, gain_diff, corr, n_sel)
end

println("\nDone. As rank approaches n=20, RSVD-OCS converges to the full-dense\n" *
        "solution -- the same pattern seen at full scale (n=5525) in\n" *
        "spruce/scripts/, just too small here to show a wall-clock speedup\n" *
        "(a 20x20 dense solve is already instant).")
