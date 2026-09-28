# ==============================================================================
# Generate the synthetic toy example: 20 individuals, 100 markers
#
# Produces a self-contained, schema-identical miniature of the real Norway
# spruce dataset so anyone can clone the repo and try the RSVD-OCS pipeline
# immediately, without access to the (unshared) Skogforsk research data.
#
# Running this script regenerates a fresh synthetic dataset with the same
# statistical design as the files already committed under example/data/ (a
# new random draw each run, since Julia's own RNG stream is what determines
# the exact numbers -- set `seed` below for a reproducible run of your own).
# You do not need to run it to try the pipeline: example/data/ already ships
# with generated output.
#
# What this produces, all written to example/data/ (small enough to commit):
#   Hmat_example_20ind.txt         -- genomic relationship matrix (VanRaden
#                                      2008), same raw format as the real
#                                      H-matrix file: comma-separated, no
#                                      header, column 1 = ID.
#   phenotypes_example_20ind.txt   -- same 17-column schema as the real
#                                      phenotype file (ID, Rad, Ruta, Planta,
#                                      Trial, Mum, Dad, Famly, Hjd_17, Hjd_7,
#                                      Sprant_17, Htv_17, Vit_7, Lev_7,
#                                      Lev_17, Trial_Famly, Trial_Ruta).
#   EBV_example_20ind.csv          -- GBLUP EBVs + PEV, same column layout
#                                      as AIREML_spruce.jl's own
#                                      EBV_<tag>.csv output, so
#                                      run_example_ocs.jl can load it
#                                      directly without running AI-REML.
#   true_breeding_values_20ind.csv -- the TRUE simulated breeding values,
#                                      something you never have for real
#                                      data. Useful for sanity-checking how
#                                      close the EBVs are to the truth.
#
# Simulation summary:
#   - 6 founders with independent genotypes at 100 unlinked markers
#     (Binomial(2, p_m), p_m ~ Uniform(0.1, 0.9)).
#   - 14 further individuals across two more generations, each with two
#     distinct earlier individuals as Mum/Dad and genotypes built by
#     Mendelian sampling from both parents.
#   - A VanRaden (2008) genomic relationship matrix from the realized
#     genotypes (same construction as qtlmas/scripts).
#   - A true additive genetic value per individual from 100 random marker
#     effects, scaled to sd=5; phenotype = intercept + Trial effect + true
#     breeding value + residual, calibrated to h2=0.3.
#   - EBVs from a GBLUP fit (one fixed effect: Trial; one random effect:
#     individual, ~N(0, G*sigma_a^2)) using the TRUE simulation variance
#     components directly, rather than estimating them via AI-REML -- a
#     deliberate shortcut for a fast, dependency-light quickstart.
#     AIREML_spruce.jl estimates them properly on real data; don't
#     `include()` it against this toy example -- its own run block at the
#     bottom executes on include and points at the real (unshipped)
#     spruce/data files.
#   - Hjd_7/Sprant_17/Htv_17/Vit_7/Lev_7/Lev_17 are filled with simple,
#     illustrative random values (not genetically simulated), purely so the
#     phenotype file matches the real file's column schema. Only Hjd_17,
#     Trial, Trial_Ruta and Trial_Famly are actually used by the pipeline.
#
# Usage:   julia generate_example_data.jl
# ==============================================================================

using Random, LinearAlgebra, Statistics, DataFrames, CSV, Printf

# ------------------------------------------------------------------------------
# 0. CONFIG
# ------------------------------------------------------------------------------
n            = 20      # individuals
m            = 100     # markers
n_founders   = 6
h2           = 0.3
target_sd_a  = 5.0      # sd of true breeding values
mu           = 150.0    # intercept (arbitrary units, "cm"-flavoured like Hjd_17)
trial_effect = Dict(1 => 2.0, 2 => -2.0)
seed         = 22

out_dir = joinpath(@__DIR__, "..", "data")
mkpath(out_dir)

Random.seed!(seed)

# ------------------------------------------------------------------------------
# 1. Pedigree: 6 unrelated founders + 14 individuals across two more
#    generations, each with two distinct earlier individuals as parents.
# ------------------------------------------------------------------------------
ids = collect(1:n)
mum = zeros(Int, n)
dad = zeros(Int, n)
for i in (n_founders + 1):n
    a, b = 0, 0
    while a == b
        a, b = rand(1:(i - 1)), rand(1:(i - 1))
    end
    mum[i] = ids[a]
    dad[i] = ids[b]
end

# ------------------------------------------------------------------------------
# 2. Genotypes: Binomial(2,p) founders, Mendelian sampling for progeny.
# ------------------------------------------------------------------------------
p_true = rand(m) .* 0.8 .+ 0.1   # Uniform(0.1, 0.9)
M = zeros(Int, n, m)
for j in 1:n_founders, k in 1:m
    M[j, k] = (rand() < p_true[k] ? 1 : 0) + (rand() < p_true[k] ? 1 : 0)   # Binomial(2, p)
end
for i in (n_founders + 1):n, k in 1:m
    dm = M[mum[i], k] / 2.0
    dd = M[dad[i], k] / 2.0
    M[i, k] = (rand() < dm ? 1 : 0) + (rand() < dd ? 1 : 0)
end

# ------------------------------------------------------------------------------
# 3. VanRaden (2008) genomic relationship matrix
# ------------------------------------------------------------------------------
p_hat = vec(mean(M, dims = 1)) ./ 2.0
Z = M .- 2 .* p_hat'
denom = 2 * sum(p_hat .* (1 .- p_hat))
G = (Z * Z') ./ denom
G = (G .+ G') ./ 2

# ------------------------------------------------------------------------------
# 4. True breeding values and phenotype
# ------------------------------------------------------------------------------
beta = randn(m)
a_raw = Z * beta
a_raw .-= mean(a_raw)
a_true = a_raw .* (target_sd_a / std(a_raw))
sigma_a2 = var(a_true)
sigma_e2 = sigma_a2 * (1 - h2) / h2

trial = [isodd(i) ? 1 : 2 for i in 1:n]
e = randn(n) .* sqrt(sigma_e2)
Hjd_17 = mu .+ [trial_effect[t] for t in trial] .+ a_true .+ e

# Illustrative-only extra trait columns (not genetically simulated), just to
# match the real file's schema.
Hjd_7     = round.(Hjd_17 .* 0.4 .+ randn(n) .* 2.0, digits = 1)
Sprant_17 = rand(1:9, n)
Htv_17    = round.(Hjd_17 .+ randn(n) .* 3.0, digits = 1)
Vit_7     = rand(1:9, n)
Lev_7     = rand(0:1, n)
Lev_17    = rand(0:1, n)

famly         = [mum[i] == 0 ? ids[i] : mum[i] * 100 + dad[i] for i in 1:n]
plot_in_trial = [((i - 1) % 4) + 1 for i in 1:n]      # 4 plots per trial
tree_in_plot  = [((i - 1) ÷ 4) + 1 for i in 1:n]
trial_ruta    = ["T$(trial[i])_R$(plot_in_trial[i])" for i in 1:n]
trial_famly   = ["T$(trial[i])_F$(famly[i])" for i in 1:n]

pheno = DataFrame(
    ID          = ids,
    Rad         = collect(1:n),
    Ruta        = plot_in_trial,
    Planta      = tree_in_plot,
    Trial       = trial,
    Mum         = mum,
    Dad         = dad,
    Famly       = famly,
    Hjd_17      = round.(Hjd_17, digits = 1),
    Hjd_7       = Hjd_7,
    Sprant_17   = Sprant_17,
    Htv_17      = Htv_17,
    Vit_7       = Vit_7,
    Lev_7       = Lev_7,
    Lev_17      = Lev_17,
    Trial_Famly = trial_famly,
    Trial_Ruta  = trial_ruta,
)

# ------------------------------------------------------------------------------
# 5. GBLUP EBVs + PEV, using the TRUE simulation variance components (see
#    header note -- AIREML_spruce.jl estimates these properly via AI-REML;
#    this script skips that for a fast, dependency-light quickstart).
#    Every individual is genotyped and has exactly one phenotype record, so
#    Z = I and the usual selection-index P-matrix form applies directly.
# ------------------------------------------------------------------------------
X = zeros(n, 2)
for i in 1:n
    X[i, trial[i]] = 1.0
end
V = G .* sigma_a2 .+ Matrix{Float64}(I, n, n) .* sigma_e2
Vinv = inv(V)
XtVinvX = X' * Vinv * X
P = Vinv - Vinv * X * inv(XtVinvX) * X' * Vinv
a_hat = sigma_a2 .* (G * (P * Hjd_17))
PEV = sigma_a2 .* G .- (sigma_a2^2) .* (G * P * G)
PEV = (PEV .+ PEV') ./ 2
pev_d = diag(PEV)
rel = 1 .- pev_d ./ (sigma_a2 .* diag(G))

@printf("corr(EBV, true breeding value) = %.3f, mean reliability = %.3f\n",
        cor(a_hat, a_true), mean(rel))

ebv_df = DataFrame(ID = ids, EBV = round.(a_hat, digits = 4),
                    PEV = round.(pev_d, digits = 4),
                    SEP = round.(sqrt.(max.(pev_d, 0)), digits = 4),
                    reliability = round.(rel, digits = 4), phenotyped = trues(n))

truth_df = DataFrame(ID = ids, true_breeding_value = round.(a_true, digits = 4))

# ------------------------------------------------------------------------------
# 6. Write outputs
# ------------------------------------------------------------------------------
hmat_file  = joinpath(out_dir, "Hmat_example_20ind.txt")
pheno_file = joinpath(out_dir, "phenotypes_example_20ind.txt")
ebv_file   = joinpath(out_dir, "EBV_example_20ind.csv")
truth_file = joinpath(out_dir, "true_breeding_values_20ind.csv")

open(hmat_file, "w") do io
    for i in 1:n
        row_vals = [@sprintf("%.6f", G[i, j]) for j in 1:n]
        println(io, string(ids[i], ",", join(row_vals, ",")))
    end
end

CSV.write(pheno_file, pheno)
CSV.write(ebv_file, ebv_df)
CSV.write(truth_file, truth_df)

println("Wrote:")
println("  ", hmat_file)
println("  ", pheno_file)
println("  ", ebv_file)
println("  ", truth_file)
