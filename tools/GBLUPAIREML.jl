# ==============================================================================
# GBLUP AI-REML SOLVER (With GxE Random Regression & LRT Statistical Testing)
# ==============================================================================
# This script performs mixed model analysis using the AI-REML algorithm.
# It handles missing phenotype values, multiple random effects, and 
# includes an automatic Random Regression Model (RRM) for GxE (Alpha/Beta).

using DelimitedFiles, Statistics, LinearAlgebra, Printf, Missings

"""
    run_gblup(; pheno_file, geno_file, cov_file, rand_file, save_pev, run_rrm, env_cov_idx, export_dynamic_pev)

Runs an AI-REML GBLUP analysis. Handles missing phenotypes, incorporates multiple 
random effects. If `run_rrm = true`, it treats `env_cov_idx` as a categorical factor,
automatically calculates the environmental indices, and fits a GxE Alpha/Beta model.

`export_dynamic_pev`: 
  - If FALSE: Outputs a single Standardized PEV matrix for the average environment.
  - If TRUE: Outputs 3 raw PEV matrices (Alpha, Beta, Cov) to build dynamic risk constraints in downstream RABOCS pipelines.
"""
function run_gblup(; 
    pheno_file         = "wheat_pheno.csv",
    geno_file          = "wheat_geno.csv",
    cov_file           = "wheat_env_cov.csv",
    rand_file          = nothing,
    save_pev           = true, 
    pev_outfile        = "PEV_matrix.csv",
    save_G             = true, 
    G_outfile          = "G_matrix.csv",
    run_rrm            = false,      
    env_cov_idx        = 1,          
    export_dynamic_pev = false,       
    is_clonal          = true
)

    ## 1. File I/O
    println("Loading data files...")
    y_raw = readdlm(pheno_file, ',') 
    M = readdlm(geno_file, ',')
    X_raw = readdlm(cov_file, ',')
    
    R_data = nothing
    if !isnothing(rand_file) && isfile(rand_file)
        R_data = readdlm(rand_file, ',')
        if size(R_data, 2) == 1 && all(iszero, R_data)
            println("Random effects file found, but no effects detected.")
            R_data = nothing
        else
            println("Random effects file found and loaded.")
        end
    end
    println("...Done loading data.\n")

    ## 2. Data Preprocessing
    println("Setting up model matrices...")
    
    # Process Phenotypes robustly 
    y = Array{Union{Float64, Missing}}(undef, size(y_raw))
    for i in eachindex(y_raw)
        val = y_raw[i]
        if val isa Real
            y[i] = Float64(val)
        elseif val isa AbstractString
            clean_val = uppercase(strip(val))
            y[i] = (clean_val == "NA" || clean_val == "") ? missing : parse(Float64, clean_val)
        else
            y[i] = missing
        end
    end
    n_obs = length(y)
    
    # -------------------------------------------------------------------
    # AUTOMATIC ENVIRONMENTAL INDEX CALCULATION (MUST BE DONE FIRST)
    # -------------------------------------------------------------------
    E_vec = zeros(n_obs)
    if run_rrm
        println("--> RRM Enabled: Calculating environmental indices from factor in column $env_cov_idx...")
        env_col = X_raw[:, env_cov_idx]
        unique_envs = unique(env_col)
        
        for env in unique_envs
            env_phenos = collect(skipmissing(y[env_col .== env]))
            env_mean = length(env_phenos) > 0 ? mean(env_phenos) : 0.0
            E_vec[env_col .== env] .= env_mean
        end
        
        # Center AND Scale the index
        E_vec .-= mean(E_vec)
        e_std = std(E_vec)
        if e_std > 0.0
            E_vec ./= e_std
        end
        
        # Export Mapping
        env_map = []
        for env in unique_envs
            idx = findfirst(==(env), env_col)
            push!(env_map, [env, E_vec[idx]])
        end
        writedlm("Environment_Mapping.csv", env_map, ',')
        println("    Saved physical-to-standardized environment map to: Environment_Mapping.csv")
    end

    # --- DYNAMIC Z_0 CONSTRUCTION ---
    if is_clonal
        println("-> Clonal mode enabled: Extracting unique biological genotypes...")
        M_unique = unique(M, dims=1)
        n_clones = size(M_unique, 1)
        println("    Detected $n_clones unique genotypes across $n_obs total observations.")
        
        Z0 = zeros(Float64, n_obs, n_clones)
        for i in 1:n_obs
            clone_idx = findfirst(row -> row == M[i, :], eachrow(M_unique))
            Z0[i, clone_idx] = 1.0
        end
    else
        println("-> Standard mode enabled: Assuming 1-to-1 phenotype-to-genotype mapping.")
        M_unique = M
        n_clones = n_obs
        Z0 = Matrix{Float64}(I, n_obs, n_obs)
    end
    
    # 1. Build the G Matrix (Exactly n_clones x n_clones)
    p = sum(M_unique, dims=1) ./ (n_clones * 2)
    W = M_unique .- (2 .* p)
    sum_2pq = 2 * sum(p .* (1 .- p))
    G = (W * W') ./ sum_2pq
    
    # 2. Build the Z_1 Covariate Matrix 
    Z1 = zeros(Float64, n_obs, n_clones)
    for i in 1:n_obs
        Z1[i, :] .= Z0[i, :] .* E_vec[i]
    end
    
    # 3. Construct the Kinship Variance Components for AI-REML
    if run_rrm
        K_int = Z0 * G * Z0'
        K_slp = Z1 * G * Z1'
        K_cov = (Z0 * G * Z1') .+ (Z1 * G * Z0')
        K_matrices_full = Matrix{Float64}[K_int, K_slp, K_cov]
    else
        K_int = Z0 * G * Z0'
        K_matrices_full = Matrix{Float64}[K_int]
    end

    # Process Fixed Effects (Add intercept, dummy-code factors)
    has_intercept = size(X_raw, 2) > 0 && all(X_raw[:, 1] .== 1.0)
    if !has_intercept
        X_raw = hcat(ones(n_obs, 1), X_raw)
        env_cov_idx += 1 
        println("Intercept column added.")
    end

    processed_X = X_raw[:, 1:1]
    for col_idx in 2:size(X_raw, 2)
        current_col = X_raw[:, col_idx]
        unique_levels = sort(unique(current_col))
        n_levels = length(unique_levels)

        if n_levels == 2
            baseline = 0.0 ∈ unique_levels ? 1.0 : unique_levels[1]
            processed_X = hcat(processed_X, current_col .== baseline)
        elseif n_levels > 2
            level_indicators = zeros(n_obs, n_levels - 1)
            for i in 2:n_levels
                level_indicators[:, i-1] = current_col .== unique_levels[i]
            end
            processed_X = hcat(processed_X, level_indicators)
        else
            processed_X = hcat(processed_X, current_col)
        end
    end
    X = processed_X

    # Add additional random effects
    if !isnothing(R_data)
        for i in 1:size(R_data, 2)
            rand_effect_levels = R_data[:, i]
            unique_levels = unique(rand_effect_levels)
            Z_rand = zeros(n_obs, length(unique_levels))
            for j in 1:length(unique_levels)
                Z_rand[rand_effect_levels .== unique_levels[j], j] .= 1.0
            end
            push!(K_matrices_full, Z_rand * Z_rand')
        end
        println("Found $(length(K_matrices_full) - (run_rrm ? 3 : 1)) additional random effects.")
    end

    ## 3. Filter for Missing Data
    observed_idx = vec(.!ismissing.(y))
    n_obs_filt = sum(observed_idx)
    y_obs = collect(skipmissing(y))
    X_obs = X[observed_idx, :]
    
    K_matrices_obs = [K[observed_idx, observed_idx] for K in K_matrices_full]

    ## 4. AI-REML Algorithm
    num_vc = length(K_matrices_obs) + 1
    σ_sq_vec = ones(num_vc)
    max_iter = 100
    convergence_threshold = 1e-6
    
    println("\nStarting AI-REML algorithm ($n_obs_filt observations)...")
    if run_rrm
        header = "Iter |    σ²_Int  |    σ²_Slp  |    σ_Cov   |" * join([" σ²_r$i |" for i in 1:(num_vc-5)]) * "   σ²_e   | Log-Likelihood | Change"
    else
        header = "Iter |    σ²_G    |" * join([" σ²_r$i |" for i in 1:length(K_matrices_obs)-1]) * "   σ²_e   | Log-Likelihood | Change"
    end
    println(header, "\n", "-"^length(header))

    AI = zeros(num_vc, num_vc)
    P_final = zeros(n_obs_filt, n_obs_filt)
    log_likelihood = 0.0

    for iter in 1:max_iter
        old_σ_sq_vec = copy(σ_sq_vec)
        
        V_obs = sum(σ_sq_vec[i] * K_matrices_obs[i] for i in 1:length(K_matrices_obs)) + σ_sq_vec[end] * I
        F = cholesky(Symmetric(V_obs)) 
        V_obs_inv = inv(F) 
        
        X_t_V_inv_X = X_obs' * V_obs_inv * X_obs
        F_X = cholesky(Symmetric(X_t_V_inv_X))
        
        P_obs = V_obs_inv - V_obs_inv * X_obs * inv(F_X) * X_obs' * V_obs_inv
        
        if iter == max_iter || all(abs.(old_σ_sq_vec - σ_sq_vec) .< convergence_threshold)
            P_final = P_obs
        end
        
        log_likelihood = -0.5 * (logdet(F) + logdet(F_X) + (y_obs' * P_obs * y_obs))
        
        gradients = zeros(num_vc)
        for i in 1:length(K_matrices_obs)
            gradients[i] = -0.5 * tr(P_obs * K_matrices_obs[i]) + 0.5 * (y_obs' * P_obs * K_matrices_obs[i] * P_obs * y_obs)
        end
        gradients[end] = -0.5 * tr(P_obs) + 0.5 * (y_obs' * P_obs * P_obs * y_obs)
        
        for i in 1:num_vc, j in i:num_vc
            K_i = (i <= length(K_matrices_obs)) ? K_matrices_obs[i] : I
            K_j = (j <= length(K_matrices_obs)) ? K_matrices_obs[j] : I
            AI[i,j] = 0.5 * tr(P_obs * K_i * P_obs * K_j)
            AI[j,i] = AI[i,j]
        end
        
        delta = AI \ gradients
        σ_sq_vec .+= delta
        
        # --- RRM COVARIANCE BENDING LOGIC ---
        if run_rrm
            v_int, v_slp, v_cov = σ_sq_vec[1], σ_sq_vec[2], σ_sq_vec[3]
            if v_int <= 0 || v_slp <= 0 || (v_int * v_slp - v_cov^2) <= 1e-8
                Sigma = Symmetric([v_int v_cov; v_cov v_slp])
                vals, vecs = eigen(Sigma)
                vals_bent = max.(vals, 1e-6)
                Sigma_bent = vecs * Diagonal(vals_bent) * vecs'
                
                σ_sq_vec[1] = Sigma_bent[1, 1]
                σ_sq_vec[2] = Sigma_bent[2, 2]
                σ_sq_vec[3] = Sigma_bent[1, 2]
            end
            for i in 4:length(σ_sq_vec)
                σ_sq_vec[i] = max(σ_sq_vec[i], 1e-8)
            end
        else
            σ_sq_vec = max.(σ_sq_vec, 1e-8)
        end
        
        change = sum(abs.(delta))
        
        @printf("%4d | ", iter)
        for val in σ_sq_vec @printf("%10.4f | ", val) end
        @printf("%14.4f | %e\n", log_likelihood, change)
        
        if change < convergence_threshold
            println("\nAlgorithm converged.")
            break
        end
    end

    ## 5. Final Predictions and Output (Projecting back to the clones)
    Z0_obs = Z0[observed_idx, :]
    Z1_obs = Z1[observed_idx, :]

    PEV_alpha, PEV_beta, PEV_cov = nothing, nothing, nothing
    PEV_std_alpha = nothing
    
    if run_rrm
        Cov_alpha_T = σ_sq_vec[1] * G * Z0_obs' + σ_sq_vec[3] * G * Z1_obs'
        Cov_beta_T  = σ_sq_vec[3] * G * Z0_obs' + σ_sq_vec[2] * G * Z1_obs'
        
        g_hat_alpha = Cov_alpha_T * (P_final * y_obs)
        g_hat_beta  = Cov_beta_T  * (P_final * y_obs)
        g_hat_all   = g_hat_alpha
        
        PEV_alpha = σ_sq_vec[1] * G - Cov_alpha_T * P_final * Cov_alpha_T'
        PEV_std_alpha = PEV_alpha ./ σ_sq_vec[1] 
        
        if export_dynamic_pev
            PEV_beta = σ_sq_vec[2] * G - Cov_beta_T * P_final * Cov_beta_T'
            PEV_cov  = σ_sq_vec[3] * G - Cov_alpha_T * P_final * Cov_beta_T'
        end
    else
        Cov_G_T = σ_sq_vec[1] * G * Z0_obs'
        g_hat_all = Cov_G_T * (P_final * y_obs)
        PEV_all = σ_sq_vec[1] * G - Cov_G_T * P_final * Cov_G_T'
        PEV_std = PEV_all ./ σ_sq_vec[1]
    end

    ## 5b. Summary Statistics (Marginal)
    marginal_aic = -2 * log_likelihood + 2 * num_vc
    
    println("\nModel Fit Criteria:")
    println("--------------------")
    @printf("Restricted Log-Likelihood: %14.4f\n", log_likelihood)
    @printf("Variance Parameters (k):   %14d\n", num_vc)
    @printf("Marginal AIC:              %14.4f\n", marginal_aic)

    ## 6. Export Results (Strictly formatted for RABOCS Master Pipeline)
    println("\nSaving Output Files...")
    
    if run_rrm
        alpha_beta_out = hcat(g_hat_alpha, g_hat_beta)
        writedlm("GEBV_Alpha_Beta_output.csv", alpha_beta_out, ',')
        println("-> GEBV (Alpha & Beta) saved to: GEBV_Alpha_Beta_output.csv")
    else
        writedlm("GEBV_output.csv", g_hat_all, ',')
        println("-> GEBV saved to: GEBV_output.csv")
    end
    
    if save_pev
        if run_rrm && export_dynamic_pev
            writedlm("PEV_Alpha.csv", PEV_alpha, ',')
            writedlm("PEV_Beta.csv", PEV_beta, ',')
            writedlm("PEV_Cov.csv", PEV_cov, ',')
            println("-> Exported 3 RAW dynamic PEV matrices (Alpha, Beta, Cov) for RABOCS Master Pipeline.")
        else
            writedlm(pev_outfile, run_rrm ? PEV_std_alpha : PEV_std, ',')
            println("-> STANDARDIZED PEV matrix successfully saved to: $pev_outfile")
        end
    end

    if save_G
        writedlm(G_outfile, G, ',')
        println("-> G matrix successfully saved to: $G_outfile")
    end
    
    println("Analysis Complete!")
    
    # NOTE: Returns heavily updated to pass LogL and parameter counts to the LRT wrapper
    if run_rrm
        if export_dynamic_pev
            return g_hat_alpha, g_hat_beta, (PEV_alpha, PEV_beta, PEV_cov), σ_sq_vec, log_likelihood, num_vc
        else
            return g_hat_alpha, g_hat_beta, PEV_std_alpha, σ_sq_vec, log_likelihood, num_vc
        end
    else
        return g_hat_all, PEV_std, σ_sq_vec, log_likelihood, num_vc
    end
end

# ==============================================================================
# STATISTICAL MODEL COMPARISON WRAPPER (LRT & Marginal AIC)
# ==============================================================================
"""
    compare_gxe_models(; pheno_file, geno_file, cov_file, env_cov_idx, is_clonal)

Runs both the Null Model (Standard GBLUP without GxE) and the Full Model (RRM with GxE).
Extracts the Log-Likelihoods to perform a formal Likelihood Ratio Test (LRT) 
to statistically prove the significance of the GxE interaction.
"""
function compare_gxe_models(; 
    pheno_file  = "wheat_pheno.csv", 
    geno_file   = "wheat_geno.csv", 
    cov_file    = "wheat_env_cov.csv", 
    env_cov_idx = 1, 
    is_clonal   = true
)
    println("\n==================================================")
    println(" 1. FITTING NULL MODEL (Baseline GBLUP No GxE)")
    println("==================================================")
    _, _, _, logL_null, k_null = run_gblup(
        pheno_file=pheno_file, geno_file=geno_file, cov_file=cov_file,
        run_rrm=false, save_pev=false, save_G=false, is_clonal=is_clonal
    )

    println("\n==================================================")
    println(" 2. FITTING FULL MODEL (RRM with Alpha/Beta)")
    println("==================================================")
    _, _, _, _, logL_full, k_full = run_gblup(
        pheno_file=pheno_file, geno_file=geno_file, cov_file=cov_file,
        run_rrm=true, env_cov_idx=env_cov_idx,
        save_pev=false, save_G=false, export_dynamic_pev=false, is_clonal=is_clonal
    )

    println("\n==================================================")
    println(" MODEL COMPARISON SUMMARY (LRT & Marginal AIC)")
    println("==================================================")
    
    # Calculate Marginal AIC (AIC = -2ln(L) + 2k)
    aic_null = -2 * logL_null + 2 * k_null
    aic_full = -2 * logL_full + 2 * k_full
    
    @printf("Null Model (No GxE): LogL = %10.4f | AIC = %10.4f (k=%d)\n", logL_null, aic_null, k_null)
    @printf("Full RRM (With GxE): LogL = %10.4f | AIC = %10.4f (k=%d)\n", logL_full, aic_full, k_full)
    
    # Likelihood Ratio Test
    lrt_stat = 2 * (logL_full - logL_null)
    df = k_full - k_null
    
    println("--------------------------------------------------")
    @printf("Likelihood Ratio Test (χ²): %10.4f\n", lrt_stat)
    @printf("Degrees of Freedom (df):    %10d\n", df)
    
    # Chi-Square Thresholds (1-tailed) for df=2
    crit_05 = 5.991
    crit_01 = 9.210
    crit_001 = 13.816
    
    if lrt_stat > crit_001
        println("Significance: p < 0.001 ***")
        println("\nConclusion: The Full RRM provides a decisively superior fit.")
        println("            The GxE interaction is highly significant and")
        println("            statistically justifies the RABOCS pipeline.")
    elseif lrt_stat > crit_05
        println("Significance: p < 0.05 *")
        println("\nConclusion: The Full RRM provides a significantly better fit.")
    else
        println("Significance: Not Significant")
        println("\nConclusion: The Null Model is sufficient. Adding the GxE")
        println("            interaction does not significantly improve fit.")
    end
    println("==================================================\n")
end

# ==============================================================================
# EXECUTION COMMANDS
# ==============================================================================



# 1. Run the Full Pipeline Prep (Generates ALL files needed for RABOCS Master Pipeline)
# EXECUTION COMMANDS  
if abspath(PROGRAM_FILE) == @__FILE__
    g_alpha, g_beta, pev_matrices, var_comps, logL, k = run_gblup(
        pheno_file         = "wheat_pheno.csv",
        geno_file          = "wheat_geno.csv",
        cov_file           = "wheat_env_cov.csv",
        run_rrm            = true,      
        env_cov_idx        = 1,         
        save_pev           = true,
        save_G             = true,
        export_dynamic_pev = true,
        is_clonal          = true
    )
end

# 2. Run Statistical Comparison (Null vs Full RRM to prove GxE exists)
#compare_gxe_models(pheno_file  = "wheat_pheno.csv", 
#    geno_file   = "wheat_geno.csv", 
#    cov_file    = "wheat_env_cov.csv", 
#    env_cov_idx = 1, 
#    is_clonal   = true)

