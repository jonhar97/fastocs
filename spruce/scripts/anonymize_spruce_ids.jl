# ==============================================================================
# Anonymize spruce IDs
#
# Replaces real tree IDs -- and the Dad / Mum pedigree links that point
# to other trees -- with dummy IDs across the two raw spruce input files:
#
#   spruce/data/Hmat_5525_spruce_tau_1_omega_1_PDF.txt   (H-matrix; col 1 = ID)
#   spruce/data/phenotypes_5525_spruce_Horder_v3.txt     (phenotypes; ID,
#                                                          Dad, Mum, ...)
#
# What this does and does not touch:
#   - The H-matrix's ROW/COLUMN ORDER and all relationship values are left
#     completely untouched. Only the ID label in column 1 of each row is
#     relabelled -- the matrix itself is not reordered or modified.
#   - Every phenotype column other than ID/Dad/Mum (Trial, Trial_Ruta,
#     Trial_Famly, Hjd_17, ...) is copied through unchanged.
#   - The SAME real-ID -> dummy-ID mapping is applied to both files and to
#     both parent columns, so a tree's own ID and its appearances as someone
#     else's Dad/Mum all map to the same dummy ID -- pedigree structure
#     is fully preserved, just relabelled.
#   - Dummy IDs are assigned in a RANDOMLY SHUFFLED order (not file order,
#     not sorted order), so a dummy ID carries no information about a tree's
#     position in the original file or anything encoded in the real ID
#     (e.g. birth year/cohort).
#   - Unknown/founder parent codes (see `missing_parent_codes` below) are
#     left exactly as they are -- they are not real tree IDs and are never
#     assigned a dummy ID.
#
# Because AIREML_spruce.jl and the rest of the pipeline only ever read the ID
# column out of these two files, you can point the pipeline at the *_anon
# files instead of the originals and every downstream output (EBVs, PEV,
# figures, CSVs) will already carry dummy IDs -- nothing downstream needs to
# change.
#
# The mapping table (real ID <-> dummy ID) is written to a separate CSV next
# to the data so the anonymization can be reversed later if you ever need to
# trace a dummy ID back to a real tree. That file is exactly as sensitive as
# the real IDs themselves -- never commit it, never share it. The repo's
# .gitignore already excludes everything under spruce/data/ (and *.csv/*.txt
# generally), so this happens automatically, but treat it with the same care
# as raw data. Set `write_mapping = false` below for one-way, irreversible
# anonymization instead.
#
# Usage:   julia anonymize_spruce_ids.jl
# ==============================================================================

using CSV, DataFrames, Random

# ------------------------------------------------------------------------------
# 0. CONFIG -- check these against your actual files before running
# ------------------------------------------------------------------------------
base      = joinpath(@__DIR__, "..")
hmat_in   = joinpath(base, "data", "Hmat_5525_spruce_tau_1_omega_1_PDF.txt")
pheno_in  = joinpath(base, "data", "phenotypes_5525_spruce_Horder_v3.txt")

hmat_out    = joinpath(base, "data", "Hmat_5525_spruce_tau_1_omega_1_PDF_anon.txt")
pheno_out   = joinpath(base, "data", "phenotypes_5525_spruce_Horder_v3_anon.txt")
mapping_out = joinpath(base, "data", "id_mapping_DO_NOT_SHARE.csv")

id_col  = "ID"
dad_col = "Dad"
mum_col = "Mum"

# Values that mean "unknown/founder parent" -- never assigned a dummy ID.
# Add any sentinel your data actually uses (e.g. "-9", "9999999", ...).
missing_parent_codes = Set(["", "0", "NA", "na", "missing"])

# Dummy IDs are plain numbers (dummy_id_base + a shuffled index), e.g.
# 900000001, 900000002, ... -- NOT a "T..."-prefixed string. This matters:
# load_spruce_hblup.jl and the rest of the _hblup pipeline read the H-matrix
# ID column with `readdlm(..., Float64)` and round it to an Int, so a
# non-numeric ID would error as soon as you tried to run those scripts on
# the anonymized files (AIREML_spruce.jl itself is fine with string IDs --
# it's specifically the _hblup scripts downstream that require numeric
# ones). Pick a base clearly outside your real ID range so dummy and real
# IDs can never collide.
dummy_id_base = 900_000_000
id_digits     = 9          # for reference only -- doesn't affect the ID value
write_mapping = true       # false => discard the mapping (one-way, irreversible)
seed          = 20260927   # fixed seed -> identical mapping every time you rerun
# ------------------------------------------------------------------------------

Random.seed!(seed)

"""
    normalize_id(s)

Mirrors AIREML_spruce.jl's own `normalize_id`: IDs written as floats (e.g.
"7.80876e6", "7808758.0") are turned into integer strings ("7808758");
anything non-numeric (e.g. "IND_1613") is kept as is. Using the identical
rule here guarantees IDs line up between the two files exactly as they do
for the real pipeline.
"""
function normalize_id(s)
    s = strip(string(s))
    x = tryparse(Float64, s)
    (x !== nothing && isfinite(x) && isinteger(x)) ? string(Int(x)) : String(s)
end

# ------------------------------------------------------------------------------
# 1. Read the H-matrix (comma-separated, no header, col 1 = ID)
# ------------------------------------------------------------------------------
println("Reading H-matrix: $hmat_in")
lines = readlines(hmat_in)
n = length(lines)
rows = Vector{Vector{SubString{String}}}(undef, n)
hmat_ids = Vector{String}(undef, n)
for (i, ln) in enumerate(lines)
    f = split(ln, ',')
    length(f) == n + 1 || error("Row $i of $hmat_in has $(length(f)-1) values, expected $n")
    rows[i] = f
    hmat_ids[i] = normalize_id(f[1])
end

# ------------------------------------------------------------------------------
# 2. Read phenotypes
# ------------------------------------------------------------------------------
println("Reading phenotypes: $pheno_in")
ph = CSV.read(pheno_in, DataFrame; missingstring = ["NA", ""],
              types = Dict(id_col => String, dad_col => String, mum_col => String))
ph[!, id_col]  = normalize_id.(coalesce.(ph[!, id_col], ""))
ph[!, dad_col] = normalize_id.(coalesce.(ph[!, dad_col], ""))
ph[!, mum_col] = normalize_id.(coalesce.(ph[!, mum_col], ""))

# ------------------------------------------------------------------------------
# 3. Build the master ID set (every ID appearing anywhere) and shuffle a dummy
#    label onto each one. Founder/unknown-parent codes are excluded.
# ------------------------------------------------------------------------------
all_real_ids = Set{String}()
union!(all_real_ids, hmat_ids)
union!(all_real_ids, ph[!, id_col])
union!(all_real_ids, ph[!, dad_col])
union!(all_real_ids, ph[!, mum_col])
setdiff!(all_real_ids, missing_parent_codes)

real_ids_shuffled = shuffle(collect(all_real_ids))
id_map = Dict(real => string(dummy_id_base + i)
              for (i, real) in enumerate(real_ids_shuffled))

dummy_for(id) = get(id_map, id, id)   # unknown-parent codes pass through unchanged

println("Anonymizing $(length(id_map)) unique IDs " *
        "(e.g. $(first(real_ids_shuffled)) -> $(dummy_for(first(real_ids_shuffled))))")

# ------------------------------------------------------------------------------
# 4. Write the anonymized H-matrix -- same row/column order, values untouched.
# ------------------------------------------------------------------------------
println("Writing anonymized H-matrix: $hmat_out")
open(hmat_out, "w") do io
    for (i, f) in enumerate(rows)
        f2 = copy(f)
        f2[1] = dummy_for(hmat_ids[i])
        println(io, join(f2, ','))
    end
end

# ------------------------------------------------------------------------------
# 5. Write the anonymized phenotype file
# ------------------------------------------------------------------------------
println("Writing anonymized phenotypes: $pheno_out")
ph2 = copy(ph)
ph2[!, id_col]  = dummy_for.(ph2[!, id_col])
ph2[!, dad_col] = dummy_for.(ph2[!, dad_col])
ph2[!, mum_col] = dummy_for.(ph2[!, mum_col])
CSV.write(pheno_out, ph2)

# ------------------------------------------------------------------------------
# 6. Save (or discard) the mapping. If kept, this file is as sensitive as the
#    real IDs -- local only, never commit/share (see header note above).
# ------------------------------------------------------------------------------
if write_mapping
    println("Writing ID mapping (local only -- do not share/commit): $mapping_out")
    CSV.write(mapping_out, DataFrame(real_id = collect(keys(id_map)),
                                      dummy_id = collect(values(id_map))))
else
    println("write_mapping = false -- anonymization is one-way, no mapping saved.")
end

println("Done. $(length(id_map)) IDs anonymized. " *
        "Original files left untouched: $hmat_in, $pheno_in")
