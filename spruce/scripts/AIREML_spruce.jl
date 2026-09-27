# ==============================================================================
# AI-REML (single trait) for the Norway spruce case
# ==============================================================================
# Model:  y = Xb + Z_p p + Z_f f + a + e
#   fixed : intercept + Trial
#   random: p ~ N(0, I σ²_p)   plot within trial   (Trial_Ruta)
#           f ~ N(0, I σ²_f)   family within trial (Trial_Famly, GxE)
#           a ~ N(0, H σ²_a)   individual additive genetic effect
#           e ~ N(0, I σ²_e)
#
# V-based AI-REML (n_obs ≈ 4,500 so dense V is cheap). The average information
# uses AI_ij = ½ (K_i Py)' P (K_j Py), with K_a = H_oo, K_p = Z_p Z_p', K_f = Z_f Z_f', K_e = I.
# Outputs EBVs, PEV (full matrix + diagonal), reliabilities, variance components with SEs.
#
# Usage:  include("AIREML_spruce.jl")   (runs the analysis block at the bottom)
#     or  julia AIREML_spruce.jl
# ==============================================================================

using CSV, DataFrames, LinearAlgebra, Statistics, Printf, DelimitedFiles, JLD2

# ------------------------------------------------------------------------------
# I/O helpers
# ------------------------------------------------------------------------------
"""
    normalize_id(s)

IDs written by Julia/CSV as floats (e.g. "7.80876e6", "7808758.0") are turned into
integer strings ("7808760"); anything non-numeric (e.g. "IND_1613") is kept as is.
"""
function normalize_id(s)
    s = strip(string(s))
    x = tryparse(Float64, s)
    (x !== nothing && isfinite(x) && isinteger(x)) ? string(Int(x)) : String(s)
end

"""
    read_relmat(file) -> (ids::Vector{String}, H::Matrix{Float64})

Reads a comma-separated relationship matrix without header, first column = ID.
"""
function read_relmat(file)
    lines = readlines(file)
    n = length(lines)
    ids = Vector{String}(undef, n)
    H = Matrix{Float64}(undef, n, n)
    for (i, ln) in enumerate(lines)
        f = split(ln, ',')
        length(f) == n + 1 || error("Row $i of $file has $(length(f)-1) values, expected $n")
        ids[i] = normalize_id(f[1])
        @inbounds for j in 1:n
            H[i, j] = parse(Float64, f[j+1])
        end
    end
    maximum(abs.(H .- H')) < 1e-8 || @warn "Relationship matrix is not symmetric; symmetrising."
    H .= (H .+ H') ./ 2
    return ids, H
end

"Index lists for each level of a factor (only levels present in `v`)."
function group_index(v)
    levels = unique(v)
    lut = Dict(l => k for (k, l) in enumerate(levels))
    idx = [Int[] for _ in levels]
    for (i, x) in enumerate(v)
        push!(idx[lut[x]], i)
    end
    return string.(levels), idx
end

"Z Z' * v for a grouping (sum within group, broadcast back)."
function zzt_mul(groups, v)
    out = zeros(length(v))
    for g in groups
        s = sum(@view v[g])
        out[g] .= s
    end
    return out
end

"tr(P Z Z') = Σ_groups Σ_{i,j∈g} P_ij"
zzt_trace(P, groups) = sum(sum(@view P[g, g]) for g in groups)

"Add σ to V[g,g] for every group (V += σ Z Z')."
function add_zzt!(V, groups, σ)
    for g in groups
        @views V[g, g] .+= σ
    end
    return V
end

# ------------------------------------------------------------------------------
# Main routine
# ------------------------------------------------------------------------------
function run_aireml_spruce(;
    pheno_file,
    rel_file,
    out_dir        = ".",
    trait          = "Hjd_17",
    id_col         = "ID",
    fixed_factor   = "Trial",
    plot_col       = "Trial_Ruta",
    family_col     = "Trial_Famly",
    max_iter       = 100,
    tol            = 1e-6,
    save_full_pev  = true,
    tag            = trait,
)
    t0 = time()
    mkpath(out_dir)

    ## 1. Data --------------------------------------------------------------
    println("Reading phenotypes: $pheno_file")
    ph = CSV.read(pheno_file, DataFrame; missingstring = ["NA", ""],
                  types = Dict(id_col => String, fixed_factor => String,
                               plot_col => String, family_col => String))
    println("Reading relationship matrix: $rel_file")
    ph[!, id_col] = normalize_id.(ph[!, id_col])
    ids, H = read_relmat(rel_file)
    N = length(ids)
    hpos = Dict(id => k for (k, id) in enumerate(ids))

    allunique(ph[!, id_col]) || error("Duplicate IDs in phenotype file (model assumes one record per tree).")
    miss = setdiff(ph[!, id_col], ids)
    isempty(miss) || error("$(length(miss)) phenotyped IDs are not in the relationship matrix, e.g. $(first(miss, 3))")

    keep = .!ismissing.(ph[!, trait])
    d = ph[keep, :]
    n = nrow(d)
    obs = [hpos[i] for i in d[!, id_col]]          # rows of H for each record

    y_raw = Float64.(d[!, trait])
    μy, sy = mean(y_raw), std(y_raw)
    y = (y_raw .- μy) ./ sy                        # standardised for numerical stability

    trial_lv, trial_g = group_index(d[!, fixed_factor])
    X = hcat(ones(n), reduce(hcat, [Float64.(d[!, fixed_factor] .== l) for l in trial_lv[2:end]];
                             init = zeros(n, 0)))
    plot_lv, plot_g = group_index(d[!, plot_col])
    fam_lv,  fam_g  = group_index(d[!, family_col])

    Hoo = H[obs, obs]

    @printf("Trait %s: %d records (of %d), mean %.2f, sd %.2f\n", trait, n, nrow(ph), μy, sy)
    @printf("Levels: %s = %d, %s = %d, %s = %d; H is %d × %d\n",
            fixed_factor, length(trial_lv), plot_col, length(plot_lv),
            family_col, length(fam_lv), N, N)

    ## 2. AI-REML ------------------------------------------------------------
    names_vc = ["σ²_a", "σ²_plot", "σ²_fam×trial", "σ²_e"]
    θ = [0.30, 0.05, 0.05, 0.60]                    # on standardised scale (var(y)=1)
    nvc = length(θ)

    V  = Matrix{Float64}(undef, n, n)
    P  = Matrix{Float64}(undef, n, n)
    AI = zeros(nvc, nvc)
    logL = -Inf
    converged = false

    function build_P!(θ)
        V .= θ[1] .* Hoo
        add_zzt!(V, plot_g, θ[2])
        add_zzt!(V, fam_g,  θ[3])
        @inbounds for i in 1:n
            V[i, i] += θ[4]
        end
        F  = cholesky!(Symmetric(V))                # V overwritten by its factor
        Vi = inv(F)
        ViX = Vi * X
        XViX = Symmetric(X' * ViX)
        FX = cholesky(XViX)
        P .= Vi .- ViX * (FX \ ViX')
        Pyv = P * y
        ll = -0.5 * (logdet(F) + logdet(FX) + dot(y, Pyv))
        return Pyv, ll
    end

    println("\nIter |" * join([@sprintf(" %12s |", s) for s in names_vc]) * "     logL (std) |  max |Δθ|/θ")
    for iter in 1:max_iter
        Py, ll = build_P!(θ)
        logL = ll

        # K_i * Py
        W = hcat(Hoo * Py, zzt_mul(plot_g, Py), zzt_mul(fam_g, Py), Py)

        trPK = [dot(P, Hoo), zzt_trace(P, plot_g), zzt_trace(P, fam_g), tr(P)]
        grad = -0.5 .* trPK .+ 0.5 .* (W' * Py)
        AI = 0.5 .* (W' * (P * W))
        AI = (AI + AI') ./ 2

        Δ = AI \ grad
        # step-halving to keep all components positive
        step = 1.0
        while any(θ .+ step .* Δ .<= 0) && step > 1e-3
            step /= 2
        end
        θnew = max.(θ .+ step .* Δ, 1e-8)
        relchg = maximum(abs.(θnew .- θ) ./ max.(θ, 1e-8))
        θ = θnew

        @printf("%4d |", iter)
        for v in θ; @printf(" %12.4f |", v * sy^2); end
        @printf(" %14.4f | %11.3e%s\n", logL, relchg, step < 1 ? "  (step $(step))" : "")

        if relchg < tol
            converged = true
            break
        end
    end
    converged || @warn "AI-REML did not converge in $max_iter iterations"

    # Final evaluation at converged θ
    Py, logL = build_P!(θ)
    W = hcat(Hoo * Py, zzt_mul(plot_g, Py), zzt_mul(fam_g, Py), Py)
    AI = 0.5 .* (W' * (P * W)); AI = (AI + AI') ./ 2
    covθ = inv(AI)

    ## 3. Back-transform & summary -------------------------------------------
    s2 = sy^2
    σ  = θ .* s2
    covσ = covθ .* s2^2
    seσ = sqrt.(max.(diag(covσ), 0))

    σP = sum(σ)
    h2 = σ[1] / σP
    g  = [(σP - σ[1]) / σP^2, -σ[1] / σP^2, -σ[1] / σP^2, -σ[1] / σP^2]   # delta method
    se_h2 = sqrt(max(g' * covσ * g, 0))
    logL_orig = logL - (n - size(X, 2)) * log(sy)  # REML logL on original scale (up to a constant)

    println("\nVariance components (original scale):")
    for k in 1:nvc
        @printf("  %-14s %12.3f  (SE %9.3f)   %5.1f%% of σ²_P\n", names_vc[k], σ[k], seσ[k], 100σ[k] / σP)
    end
    @printf("  h² (tree)      %12.3f  (SE %9.3f)\n", h2, se_h2)
    @printf("  REML logL      %12.4f   converged = %s\n", logL_orig, converged)

    ## 4. BLUEs / BLUPs ---------------------------------------------------------
    # BLUE of b: y − X b̂ = V P y, so b̂ = X \ (y − V P y)
    Vfull = θ[1] .* Hoo; add_zzt!(Vfull, plot_g, θ[2]); add_zzt!(Vfull, fam_g, θ[3])
    Vfull[diagind(Vfull)] .+= θ[4]
    resid_fixed = Vfull * Py                       # = y - X b̂
    b_std = X \ (y .- resid_fixed)
    Vfull = nothing
    b = b_std .* sy; b[1] += μy
    println("\nFixed effects (intercept = level $(trial_lv[1])):")
    @printf("  %-22s %10.3f\n", "intercept", b[1])
    for (k, l) in enumerate(trial_lv[2:end]); @printf("  %-22s %10.3f\n", "Trial " * l, b[k+1]); end

    # Additive genetic values for all N individuals in H: a_hat = σ²_a H[:,obs] P y
    B  = H[:, obs]
    a_hat  = θ[1] .* (B * Py) .* sy
    p_hat  = [θ[2] * sum(Py[gi]) for gi in plot_g] .* sy
    f_hat  = [θ[3] * sum(Py[gi]) for gi in fam_g] .* sy

    # PEV(a_hat) = σ²_a H − σ⁴_a H[:,obs] P H[obs,:]   (original scale)
    println("\nComputing PEV matrix ($N × $N)...")
    BP  = B * P
    PEV = θ[1] .* H .- θ[1]^2 .* (BP * B')
    PEV .*= s2
    PEV .= (PEV .+ PEV') ./ 2
    BP = nothing; B = nothing

    pev_d = diag(PEV)
    rel   = 1 .- pev_d ./ (σ[1] .* diag(H))
    phenotyped = falses(N); phenotyped[obs] .= true

    @printf("  Mean reliability: phenotyped %.3f, unphenotyped %.3f\n",
            mean(rel[phenotyped]), any(.!phenotyped) ? mean(rel[.!phenotyped]) : NaN)

    ## 5. Save ------------------------------------------------------------------
    ebv = DataFrame(ID = ids, EBV = a_hat, PEV = pev_d, SEP = sqrt.(max.(pev_d, 0)),
                    reliability = rel, phenotyped = phenotyped)
    CSV.write(joinpath(out_dir, "EBV_$(tag).csv"), ebv)

    vc = DataFrame(component = [names_vc; "h2"], estimate = [σ; h2], SE = [seσ; se_h2])
    CSV.write(joinpath(out_dir, "varcomp_$(tag).csv"), vc)

    CSV.write(joinpath(out_dir, "plot_effects_$(tag).csv"), DataFrame(Trial_Ruta = plot_lv, BLUP = p_hat))
    CSV.write(joinpath(out_dir, "famxtrial_effects_$(tag).csv"), DataFrame(Trial_Famly = fam_lv, BLUP = f_hat))

    if save_full_pev
        jldsave(joinpath(out_dir, "AIREML_$(tag).jld2");
                ids, ebv = a_hat, PEV, sigma = σ, sigma_names = names_vc, cov_sigma = covσ,
                h2, se_h2, fixed = b, trial_levels = trial_lv, logL = logL_orig)
        println("  Saved full PEV matrix to AIREML_$(tag).jld2 (key \"PEV\", rows/cols ordered as `ids`)")
    end
    @printf("\nDone in %.1f s. Results in %s\n", time() - t0, abspath(out_dir))

    return (ids = ids, ebv = a_hat, PEV = PEV, sigma = σ, se_sigma = seσ, h2 = h2,
            se_h2 = se_h2, fixed = b, logL = logL_orig, converged = converged)
end

# ==============================================================================
# Run
# ==============================================================================
if abspath(PROGRAM_FILE) == @__FILE__() || isinteractive()
    base = joinpath(@__DIR__, "..")
    res = run_aireml_spruce(
        pheno_file = joinpath(base, "data", "phenotypes_5525_spruce_Horder_v3.txt"),
        rel_file   = joinpath(base, "data", "Hmat_5525_spruce_tau_1_omega_1_PDF.txt"),
        # anonymised pedigree-only alternative:
        # pheno_file = joinpath(base, "data", "phenotypes_5525_spruce_anon.txt"),
        # rel_file   = joinpath(base, "data", "JWAS_A_5525_anon.txt"),
        out_dir    = joinpath(base, "output"),
        trait      = "Hjd_17",
    )
end
