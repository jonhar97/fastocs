## build_robust_omega.jl
##
## Reads the JWAS MCMC posterior-sample files for the three traits behind
## the Norway spruce selection index (Hjd17 + Htv17 - Sprant17)/3, and
## builds the centred posterior sample matrix X and its exact low-rank
## factor F_Ω = X/√(S-1), such that Ω̂ = F_Ω F_Ω' reproduces the posterior
## sample covariance of the *index* exactly (Eq. "MCMC posterior samples"
## in the Robust OCS Methods addition). Optionally also returns a
## further rank-compressed factor via randomized SVD of X itself.
##
## ASSUMED FILE LAYOUT (JWAS default for MCMC_samples_EBV_*.txt):
##   - comma-delimited
##   - header row = individual IDs, one column per individual
##   - each subsequent row = one retained post-burn-in MCMC draw
##   i.e. each file is S rows x n columns.
## This has NOT been verified against your actual files (no local access
## in this environment). Run `inspect_mcmc_file` on one file first and
## confirm the printed columns really are individual IDs, not e.g. an
## iteration index or a "Trait" label column, before trusting the rest
## of the pipeline. If the layout differs, the fix is almost certainly
## just in read_jwas_mcmc_samples below.
##
## Usage:
##   include("build_robust_omega.jl")
##   inspect_mcmc_file(joinpath(RESULTS_DIR, TRAIT_FILES.Hjd17))   # sanity check first
##   result = build_omega_factor(RESULTS_DIR; compress_rank = 30)
##   save_omega_factor(result, joinpath(RESULTS_DIR, "omega_factor.jld2"))

using CSV, DataFrames, LinearAlgebra, Random, JLD2, Statistics

RESULTS_DIR = raw"C:\Users\JOAH\OneDrive - Skogforsk\Documents\Projekt\Optimum contribution selection\NorwaySpruceData\results_JWAS_5525_H_tau1_omega1"

TRAIT_FILES = (
    Hjd17    = "MCMC_samples_EBV_Hjd17.txt",
    Htv17    = "MCMC_samples_EBV_Htv17.txt",
    Sprant17 = "MCMC_samples_EBV_Sprant17.txt",
)

# Selection index weights, matching (Hjd17 + Htv17 - Sprant17)/3.
# Named OMEGA_INDEX_WEIGHTS (not INDEX_WEIGHTS) and left as a plain
# (non-const) assignment deliberately: gamma_rank_sweep_spruce.jl and
# spruce_single_gamma_comparison.jl each define their own INDEX_WEIGHTS
# global (lowercase fields, no `const`) for the same purpose. Reusing
# that exact name here previously caused
# "ERROR: invalid redefinition of constant Main.INDEX_WEIGHTS" when
# both scripts were include()'d in the same session -- this avoids it
# rather than relying on include order.
OMEGA_INDEX_WEIGHTS = (Hjd17 = 1/3, Htv17 = 1/3, Sprant17 = -1/3)


"""
    inspect_mcmc_file(path; n_preview=5)

Print the detected column count/names and a small preview of an MCMC
sample file, so you can confirm the header row is really individual IDs
(and not e.g. an iteration counter) before running the full pipeline.
Run this once per file before trusting `build_omega_factor`.
"""
function inspect_mcmc_file(path::AbstractString; n_preview::Int = 5)
    isfile(path) || error("File not found: $path")
    df = CSV.read(path, DataFrame; header = 1, normalizenames = false, limit = 5)
    println("File: ", path)
    println("  Columns detected (first $n_preview of $(ncol(df))): ",
            first(names(df), min(n_preview, ncol(df))))
    println("  First few rows x first $n_preview columns:")
    show(df[:, 1:min(n_preview, ncol(df))])
    println()
    println("  (Re-run with the full file to get the true row count S; ",
            "`limit=5` above only previews the header + a few rows.)")
    return df
end


"""
    read_jwas_mcmc_samples(path)

Read a JWAS `MCMC_samples_EBV_*.txt` file. Returns
(ids::Vector{String}, samples::Matrix{Float64}) where `samples` is
S × n (rows = MCMC draws, columns = individuals, in header order).
"""
function read_jwas_mcmc_samples(path::AbstractString)
    isfile(path) || error("File not found: $path")
    df = CSV.read(path, DataFrame; header = 1, normalizenames = false)
    ids = string.(names(df))
    samples = Float64.(Matrix(df))
    return ids, samples
end


"""
    align_and_combine(files, weights, dir)

Read all trait MCMC sample files, verify their individual-ID columns are
identical and identically ordered (checked rather than assumed — a
single multi-trait JWAS run should preserve this, but a re-sorted or
mismatched file would silently corrupt the index otherwise), and return
the weighted selection-index sample matrix. Returns
(ids::Vector{String}, combined::Matrix{Float64}) with
combined[s, i] = weighted selection-index value for individual i at
MCMC draw s (S × n).
"""
function align_and_combine(files::NamedTuple, weights::NamedTuple, dir::AbstractString)
    trait_names = keys(files)
    ids_ref = Vector{String}()
    combined = Matrix{Float64}(undef, 0, 0)

    for (i, trait) in enumerate(trait_names)
        path = joinpath(dir, files[trait])
        ids, samples = read_jwas_mcmc_samples(path)

        if i == 1
            ids_ref = ids
            combined = zeros(size(samples))
        else
            ids == ids_ref || error(
                "Individual ID column mismatch: $(files[trait]) does not " *
                "match the ID order of $(files[trait_names[1]]). All three " *
                "trait files must come from the same JWAS run with " *
                "identical individual ordering."
            )
            size(samples) == size(combined) || error(
                "Sample dimension mismatch for $(files[trait]): expected " *
                "$(size(combined)) (S x n), got $(size(samples))."
            )
        end

        combined .+= weights[trait] .* samples
    end

    return ids_ref, combined  # S x n
end


"""
    build_omega_factor(dir=RESULTS_DIR; compress_rank=nothing, rng=Random.default_rng())

Build the centred posterior-sample matrix and its exact low-rank factor
F_Ω = X/√(S-1) for the Norway spruce selection index
(Hjd17 + Htv17 - Sprant17)/3, from the three JWAS MCMC sample files.
If `compress_rank` is given, also returns a further rank-r factor
obtained via randomized SVD of X (never forming the dense n×n Ω̂).

Returns a NamedTuple:
  ids            :: Vector{String}         individual IDs (F_Ω row order)
  ĝ              :: Vector{Float64}        posterior mean index, length n
  S              :: Int                    number of retained MCMC draws
  n              :: Int                    number of individuals
  FΩ             :: Matrix{Float64}        n × S exact factor, Ω̂ = FΩ*FΩ'
  FΩ_compressed  :: Union{Matrix{Float64},Nothing}   n × compress_rank, if requested
"""
function build_omega_factor(dir::AbstractString = RESULTS_DIR;
                             compress_rank::Union{Int,Nothing} = nothing,
                             rng = Random.default_rng())
    ids, samples_Sxn = align_and_combine(TRAIT_FILES, OMEGA_INDEX_WEIGHTS, dir)
    S, n = size(samples_Sxn)

    ĝ = vec(mean(samples_Sxn, dims = 1))    # length n, posterior mean index
    Xt = samples_Sxn .- ĝ'                   # S x n, centred
    X = permutedims(Xt)                      # n x S, centred (paper's X convention)

    FΩ = X ./ sqrt(S - 1)                    # n x S exact factor

    FΩ_compressed = compress_rank === nothing ? nothing :
                     rsvd_compress_factor(X, compress_rank; rng = rng)

    println("Built Ω factor: n = $n individuals, S = $S retained MCMC draws.")
    if compress_rank !== nothing
        println("Compressed factor rank r_Ω = $compress_rank (from S = $S).")
    end

    return (ids = ids, ĝ = ĝ, S = S, n = n, FΩ = FΩ, FΩ_compressed = FΩ_compressed)
end


"""
    rsvd_compress_factor(X, r; oversample=10, rng=Random.default_rng())

Given the centred n×S posterior-sample matrix X (with Ω̂ = XX'/(S-1)),
return a rank-r factor Fr (n×r) such that Fr*Fr' ≈ Ω̂, via randomized
SVD applied directly to X — i.e. never forming the dense n×n Ω̂ — using
the same range-finder-then-project structure as Algorithm 1 in the
manuscript (Halko et al. 2011), here applied to X rather than to G.
"""
function rsvd_compress_factor(X::AbstractMatrix{Float64}, r::Int;
                               oversample::Int = 10, rng = Random.default_rng())
    n, S = size(X)
    k = min(r + oversample, S, n)

    Rtest = randn(rng, S, k)                 # random test matrix, S x k
    Y = X * Rtest                            # n x k sketch of X's column space
    Q = Matrix(qr(Y).Q)[:, 1:k]              # n x k orthonormal basis (explicit slice
                                              # for safety across Julia versions)

    B = Q' * X                               # k x S
    F = svd(B)                               # economy SVD: F.U (k x k), F.S (length k)

    Uk = Q * F.U[:, 1:r]                     # n x r
    σk = F.S[1:r]                            # length r

    # Ω̂ ≈ (1/(S-1)) * Uk * diag(σk)^2 * Uk' = Fr * Fr' with:
    Fr = Uk .* (σk ./ sqrt(S - 1))'          # n x r, scaled factor
    return Fr
end


"""
    save_omega_factor(result, outpath)

Persist the `build_omega_factor(...)` output to a JLD2 file for reuse in
the robust OCS SQP solver, avoiding re-reading the raw MCMC sample files
each time (the exact factor FΩ alone is n×S ≈ 5525×S in size, so this
is worth caching once rather than rebuilding per solver run).
"""
function save_omega_factor(result, outpath::AbstractString)
    @save outpath ids=result.ids ĝ=result.ĝ S=result.S n=result.n FΩ=result.FΩ FΩ_compressed=result.FΩ_compressed
    println("Saved Ω factor to: ", outpath)
    return outpath
end


## ------------------------------------------------------------------
## Example usage (uncomment to run):
##
##   inspect_mcmc_file(joinpath(RESULTS_DIR, TRAIT_FILES.Hjd17))
##
##   result = build_omega_factor(RESULTS_DIR; compress_rank = 30)
##   println("n = ", result.n, ", S = ", result.S)
##   println("Exact factor size: ", size(result.FΩ))
##   println("Compressed factor size: ", size(result.FΩ_compressed))
##
##   save_omega_factor(result, joinpath(RESULTS_DIR, "omega_factor.jld2"))
## ------------------------------------------------------------------
