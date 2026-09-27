# =============================================================================
# Shared data loader for the Norway spruce OCS / robust OCS scripts
# (single-trait Hjd17, single-step HBLUP AI-REML; see AIREML_spruce.jl)
#
# Defines in the calling scope:
#   all_ids :: Vector{Int}      candidate IDs, in H's row order
#   G       :: Matrix{Float64}  H matrix restricted to candidates
#   g_vec   :: Vector{Float64}  AI-REML EBVs for Hjd17 (cm), aligned to G
#   PEV     :: Matrix{Float64}  AI-REML prediction error (co)variance matrix,
#                               aligned to G  (the Ω of robust OCS)
#   n_total :: Int
#
# Reference trees (unimproved material with genetic groups as parents) have
# no non-zero relationships in H; they are dropped from the candidate set
# only when EXCLUDE_REFERENCE = true (default false: keep all 5525 so the
# full pedigree size is used for the speedup comparison).
# =============================================================================

using DelimitedFiles, CSV, DataFrames, JLD2, LinearAlgebra, Statistics, Printf

SPRUCE_DIR   = normpath(joinpath(@__DIR__, ".."))
geno_file    = joinpath(SPRUCE_DIR, "data",   "Hmat_5525_spruce_tau_1_omega_1_PDF.txt")
ebv_file     = joinpath(SPRUCE_DIR, "output", "EBV_Hjd_17.csv")
aireml_file  = joinpath(SPRUCE_DIR, "output", "AIREML_Hjd_17.jld2")
@isdefined(EXCLUDE_REFERENCE) || (EXCLUDE_REFERENCE = false)
@isdefined(LOAD_PEV)          || (LOAD_PEV = true)

println("[1] Loading H, AI-REML EBVs (Hjd17)", LOAD_PEV ? " and PEV" : "", "...")
M = readdlm(geno_file, ',', Float64; header = false)
all_ids = Int.(round.(M[:, 1]))
G = Matrix(M[:, 2:end]); M = nothing
@assert size(G, 1) == size(G, 2) == length(all_ids)
G = (G + G') / 2

ebv = CSV.read(ebv_file, DataFrame; types = Dict(:ID => String))
id_to_g = Dict(parse(Int, ebv.ID[i]) => ebv.EBV[i] for i in 1:nrow(ebv))
@assert all(haskey(id_to_g, id) for id in all_ids) "Some H individuals have no EBV in $ebv_file"
g_vec = [id_to_g[id] for id in all_ids]

if LOAD_PEV
    pev_ids, PEV, ebv_jld = jldopen(aireml_file, "r") do f
        parse.(Int, f["ids"]), f["PEV"], f["ebv"]
    end
    pos = Dict(id => i for (i, id) in enumerate(pev_ids))
    ord = [pos[id] for id in all_ids]
    ord == 1:length(ord) || (PEV = PEV[ord, ord])
    r = cor(g_vec, ebv_jld[ord])
    r > 0.9999 || @warn "EBVs in $ebv_file and $aireml_file disagree (r=$r) -- different AI-REML runs?"
end

is_reference = vec(count(!iszero, G; dims = 2)) .== 1     # only the diagonal is non-zero
@printf("  %d individuals in H, of which %d reference trees (no relatives in H)\n",
        length(all_ids), sum(is_reference))
if EXCLUDE_REFERENCE
    keep    = .!is_reference
    all_ids = all_ids[keep]
    G       = G[keep, keep]
    g_vec   = g_vec[keep]
    LOAD_PEV && (PEV = PEV[keep, keep])
    println("  Reference trees excluded from the candidate set.")
end
n_total = length(all_ids)
@printf("  Candidates: n=%d   EBV mean=%.2f sd=%.2f range=[%.1f, %.1f] cm\n\n",
        n_total, mean(g_vec), std(g_vec), minimum(g_vec), maximum(g_vec))
GC.gc()
