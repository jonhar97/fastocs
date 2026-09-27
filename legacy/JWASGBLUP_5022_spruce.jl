# Step 1: Load packages
using JWAS,DataFrames,CSV,Statistics,JWAS.Datasets
using Bigsimr
using DelimitedFiles
using LinearAlgebra


###################### get data
#### G 1218 pedigree members
Amat_t = readdlm("C:\\Users\\joah\\OneDrive - Skogforsk\\Documents\\Projekt\\Optimum contribution selection\\NorwaySpruceData\\Save\\Hmat_5525_spruce_tau_1_omega_1.txt",',', Float64,'\n', header=false)
IDs = Amat_t[:,1]
# Remove the first column (IDs) by selecting columns 2 to end
Hmat = Amat_t[:, 2:end]

# Check if relationship matrix is positive definite
Gpd = cor_nearPD(Hmat)
isposdef(Gpd)
isposdef(Hmat)

HmatID = hcat(IDs,Gpd)
df = DataFrame(HmatID, :auto)
df.x1 = convert(Vector{Int}, df.x1)
CSV.write("C:\\Users\\joah\\OneDrive - Skogforsk\\Documents\\Projekt\\Optimum contribution selection\\NorwaySpruceData\\Save\\Hmat_5525_spruce_tau_1_omega_1_PDF.txt", df, writeheader=false)

# Step 2: Read data 
#phenofile  = dataset("phenotypes.csv")
#pedfile    = dataset("pedigree.csv")
#genofile   = dataset("GRM.csv")
phenofile  = dataset("C:\\Users\\joah\\OneDrive - Skogforsk\\Documents\\Projekt\\Optimum contribution selection\\NorwaySpruceData\\Save\\phenotypes_5525_spruce_Horder_v3.txt") 
genofile   = dataset("C:\\Users\\joah\\OneDrive - Skogforsk\\Documents\\Projekt\\Optimum contribution selection\\NorwaySpruceData\\Save\\Hmat_5525_spruce_tau_1_omega_1_PDF.txt")


phenotypes = CSV.read(phenofile,DataFrame,delim = ',',header=true,missingstrings=["NA"])
#pedigree   = get_pedigree(pedfile,separator=",",header=true);
#pedigree   = get_pedigree(pedfile,separator=",",header=true);
genotypes  = get_genotypes(genofile,separator=',',method="GBLUP",header=false);
first(phenotypes,5)

# Step 3: Transform the levels in Sprant17
#phenotypes.Sprant17 = replace(phenotypes.Sprant17, 3 => 4, 2 => 3, 1 => 2, 0 => 1, 9 => 5)
phenotypes.Lev17 = replace(phenotypes.Lev17, 1 => 2, 0 => 1)
# Verify the transformation
println(unique(phenotypes.Sprant17))

# Step 3: Build Model Equations

#model_equation  ="y1 = intercept + x1 + x2 + x2*x3 + ID + dam + genotypes"
############## traits
## Hjd_17,Hjd_07,Sprant_17,Htv_17,Vit_07                   Hjd7 = intercept + Trial + Trial_Ruta + genotypes 
model_equation  ="Hjd17 = intercept + Trial + Trial_Ruta + genotypes
                  Hjd7 = intercept + Trial + Trial_Ruta + genotypes 
                  Sprant17 = intercept + Trial + Trial_Ruta + genotypes
                  Htv17 = intercept + Trial + Trial_Ruta + genotypes
                  Lev17 = intercept + Trial + Trial_Ruta + genotypes"  # Trial*Family_id + 
model = build_model(model_equation,categorical_trait=["Lev17"]); # 

# Step 4: Set Factors or Covariates
#set_random(model,"Trial");

# Step 5: Set Random or Fixed Effects
#set_random(model,"Trial_Family");
set_random(model,"Trial_Ruta");
#set_random(model,"Ruta");
#set_random(model,"Family");
#set_random(model,"genotypes");
#set_random(model,"ID",pedigree);

# Check the structure of your data
println("Phenotype data dimensions: ", size(phenotypes))
println("First few phenotype IDs: ", first(phenotypes.ID, 5))
println("Genotype data info:")
println("Number of individuals in genotypes: ", genotypes.obsID |> length)
println("First few genotype IDs: ", first(genotypes.obsID, 5))

# Check for ID mismatches
pheno_ids = Set(phenotypes.ID)
geno_ids = Set(genotypes.obsID)

missing_in_geno = setdiff(pheno_ids, geno_ids)
missing_in_pheno = setdiff(geno_ids, pheno_ids)

println("IDs in phenotypes but not in genotypes: ", length(missing_in_geno))
println("IDs in genotypes but not in phenotypes: ", length(missing_in_pheno))

if length(missing_in_geno) > 0
    println("Some missing in genotypes: ", first(collect(missing_in_geno), min(5, length(missing_in_geno))))
end
if length(missing_in_pheno) > 0
    println("Some missing in phenotypes: ", first(collect(missing_in_pheno), min(5, length(missing_in_pheno))))
end

# Convert scientific notation IDs in genotypes to regular format
genotypes.obsID = [string(Int64(parse(Float64, id))) for id in genotypes.obsID]

# Verify the conversion worked
println("After conversion:")
println("First few genotype IDs: ", first(genotypes.obsID, 5))
println("First few phenotype IDs: ", first(phenotypes.ID, 5))

# Now check for matches
pheno_ids = Set(phenotypes.ID)
geno_ids = Set(genotypes.obsID)

missing_in_geno = setdiff(pheno_ids, geno_ids)
missing_in_pheno = setdiff(geno_ids, pheno_ids)

println("IDs in phenotypes but not in genotypes: ", length(missing_in_geno))
println("IDs in genotypes but not in phenotypes: ", length(missing_in_pheno))

# Check if we have perfect match now
common_ids = intersect(phenotypes.ID, genotypes.obsID)
println("Number of common IDs: ", length(common_ids))

# First, let's see what the genotype IDs look like after conversion attempt
println("Current genotype IDs (first 10):")
println(genotypes.obsID[1:10])
println("Type of genotype IDs: ", typeof(genotypes.obsID[1]))

println("Current phenotype IDs (first 10):")
println(phenotypes.ID[1:10])
println("Type of phenotype IDs: ", typeof(phenotypes.ID[1]))
# Step 6: Run Analysis
out=runMCMC(model,phenotypes,chain_length=1000);
print(out["heritability"])
# Step 7: Check Accuruacy

# Step 8: save output for downstream analysis
#out[]

#######################
# Calculate Posterior Genetic Correlations and Standard Deviations from JWAS Output
# Calculate Posterior Genetic Correlations and Standard Deviations from JWAS Output
using Statistics, LinearAlgebra, DataFrames, CSV

# Extract genetic variance-covariance components from JWAS output
genetic_var_df = out["genetic_variance"]
trait_names = ["Hjd17", "Hjd7", "Sprant17", "Htv17", "Lev17"]
n_traits = length(trait_names)

# Display the genetic variance components
println("Genetic Variance-Covariance Components:")
println("=" ^ 50)
println(genetic_var_df)
println()

# Create a function to extract the genetic covariance matrix
function extract_genetic_matrix(var_df, trait_names)
    n_traits = length(trait_names)
    G = zeros(n_traits, n_traits)
    
    # Create a dictionary to map covariance names to values
    cov_dict = Dict()
    for row in eachrow(var_df)
        cov_name = string(row.Covariance)
        cov_dict[cov_name] = row.Estimate
    end
    
    # Fill the genetic covariance matrix
    for i in 1:n_traits
        for j in 1:n_traits
            trait_i = trait_names[i]
            trait_j = trait_names[j]
            
            # Try both possible naming conventions
            cov_name1 = "$(trait_i)_$(trait_j)"
            cov_name2 = "$(trait_j)_$(trait_i)"
            
            if haskey(cov_dict, cov_name1)
                G[i, j] = cov_dict[cov_name1]
            elseif haskey(cov_dict, cov_name2)
                G[i, j] = cov_dict[cov_name2]
            else
                # If not found, set to 0 (might be missing covariances)
                G[i, j] = 0.0
            end
        end
    end
    
    return G
end

# Extract the genetic covariance matrix
G = extract_genetic_matrix(genetic_var_df, trait_names)

println("Genetic Covariance Matrix:")
println("=" ^ 40)
G_df = DataFrame(G, trait_names)
insertcols!(G_df, 1, :Trait => trait_names)
println(G_df)
println()

# Calculate genetic correlations
function cov_to_corr(G)
    n = size(G, 1)
    R = zeros(n, n)
    
    for i in 1:n
        for j in 1:n
            if i == j
                R[i, j] = 1.0
            else
                if G[i, i] > 0 && G[j, j] > 0
                    R[i, j] = G[i, j] / sqrt(G[i, i] * G[j, j])
                else
                    R[i, j] = 0.0
                end
            end
        end
    end
    return R
end

# Calculate genetic correlation matrix
R = cov_to_corr(G)

println("Genetic Correlation Matrix:")
println("=" ^ 40)
R_df = DataFrame(R, trait_names)
insertcols!(R_df, 1, :Trait => trait_names)
println(R_df)
println()

# Extract unique correlations and create results table
correlation_results = DataFrame(
    Trait1 = String[],
    Trait2 = String[],
    Genetic_Correlation = Float64[],
    Covariance = Float64[],
    Trait1_Variance = Float64[],
    Trait2_Variance = Float64[]
)

for i in 1:n_traits
    for j in (i+1):n_traits
        push!(correlation_results, [
            trait_names[i],
            trait_names[j],
            R[i, j],
            G[i, j],
            G[i, i],
            G[j, j]
        ])
    end
end

# Calculate standard errors using Delta method approximation
# For correlation r_ij = σ_ij / sqrt(σ_ii * σ_jj)
# SE approximation requires the standard errors of the variance components

function calculate_correlation_se(cov_ij, var_i, var_j, se_cov_ij, se_var_i, se_var_j)
    # Delta method for correlation standard error
    # This is an approximation - exact calculation would require full covariance matrix of estimates
    
    if var_i <= 0 || var_j <= 0
        return 0.0
    end
    
    r_ij = cov_ij / sqrt(var_i * var_j)
    
    # Partial derivatives
    d_r_d_cov = 1 / sqrt(var_i * var_j)
    d_r_d_var_i = -0.5 * cov_ij / (var_i * sqrt(var_i * var_j))
    d_r_d_var_j = -0.5 * cov_ij / (var_j * sqrt(var_i * var_j))
    
    # Approximate standard error (assuming independence of estimates)
    se_r = sqrt(
        (d_r_d_cov * se_cov_ij)^2 +
        (d_r_d_var_i * se_var_i)^2 +
        (d_r_d_var_j * se_var_j)^2
    )
    
    return se_r
end

# Add standard errors to results (initialize with correct length)
correlation_results[!, :SE_Correlation] = zeros(Float64, nrow(correlation_results))

# Create lookup for standard errors
se_dict = Dict()
for row in eachrow(genetic_var_df)
    cov_name = string(row.Covariance)
    se_dict[cov_name] = row.SD
end

for i in 1:nrow(correlation_results)
    trait1 = correlation_results[i, :Trait1]
    trait2 = correlation_results[i, :Trait2]
    
    # Get standard errors
    se_cov = get(se_dict, "$(trait1)_$(trait2)", get(se_dict, "$(trait2)_$(trait1)", 0.0))
    se_var1 = get(se_dict, "$(trait1)_$(trait1)", 0.0)
    se_var2 = get(se_dict, "$(trait2)_$(trait2)", 0.0)
    
    # Calculate correlation SE
    se_corr = calculate_correlation_se(
        correlation_results[i, :Covariance],
        correlation_results[i, :Trait1_Variance],
        correlation_results[i, :Trait2_Variance],
        se_cov, se_var1, se_var2
    )
    
    # Store the calculated SE
    correlation_results[i, :SE_Correlation] = se_corr
end

# Add confidence intervals (approximate)
correlation_results[!, :CI_Lower] = correlation_results[!, :Genetic_Correlation] - 1.96 * correlation_results[!, :SE_Correlation]
correlation_results[!, :CI_Upper] = correlation_results[!, :Genetic_Correlation] + 1.96 * correlation_results[!, :SE_Correlation]

# Display results
println("Genetic Correlations with Standard Errors:")
println("=" ^ 60)
for i in 1:nrow(correlation_results)
    row = correlation_results[i, :]
    println("$(row.Trait1) - $(row.Trait2):")
    println("  Correlation: $(round(row.Genetic_Correlation, digits=4)) ± $(round(row.SE_Correlation, digits=4))")
    println("  95% CI: [$(round(row.CI_Lower, digits=4)), $(round(row.CI_Upper, digits=4))]")
    println()
end

# Clean up results for final display
final_results = select(correlation_results, 
    :Trait1, :Trait2, :Genetic_Correlation, :SE_Correlation, :CI_Lower, :CI_Upper)

println("Summary Table:")
println(final_results)

# Save results
output_path = "C:\\Users\\joah\\OneDrive - Skogforsk\\Documents\\Projekt\\Optimum contribution selection\\NorwaySpruceData\\Save\\genetic_correlations_results_G_1218.csv"
CSV.write(output_path, final_results)
println("\nResults saved to 'genetic_correlations_results_G_1218.csv'")

# Also display heritabilities for reference
println("\nHeritabilities:")
println("=" ^ 30)
heritability_df = out["heritability"]
println(heritability_df)

# Calculate Posterior Genetic Correlations and Standard Deviations from JWAS Output
using Statistics, LinearAlgebra, DataFrames, CSV

# Extract genetic variance-covariance components from JWAS output
genetic_var_df = out["genetic_variance"]
trait_names = ["Hjd17", "Hjd7", "Sprant17", "Htv17", "Lev17"]
n_traits = length(trait_names)

# Display the genetic variance components
println("Genetic Variance-Covariance Components:")
println("=" ^ 50)
println(genetic_var_df)
println()

# Create a function to extract the genetic covariance matrix
function extract_genetic_matrix(var_df, trait_names)
    n_traits = length(trait_names)
    G = zeros(n_traits, n_traits)
    
    # Create a dictionary to map covariance names to values
    cov_dict = Dict()
    for row in eachrow(var_df)
        cov_name = string(row.Covariance)
        cov_dict[cov_name] = row.Estimate
    end
    
    # Fill the genetic covariance matrix
    for i in 1:n_traits
        for j in 1:n_traits
            trait_i = trait_names[i]
            trait_j = trait_names[j]
            
            # Try both possible naming conventions
            cov_name1 = "$(trait_i)_$(trait_j)"
            cov_name2 = "$(trait_j)_$(trait_i)"
            
            if haskey(cov_dict, cov_name1)
                G[i, j] = cov_dict[cov_name1]
            elseif haskey(cov_dict, cov_name2)
                G[i, j] = cov_dict[cov_name2]
            else
                # If not found, set to 0 (might be missing covariances)
                G[i, j] = 0.0
            end
        end
    end
    
    return G
end

# Extract the genetic covariance matrix
G = extract_genetic_matrix(genetic_var_df, trait_names)

println("Genetic Covariance Matrix:")
println("=" ^ 40)
G_df = DataFrame(G, trait_names)
insertcols!(G_df, 1, :Trait => trait_names)
println(G_df)
println()

# Calculate genetic correlations
function cov_to_corr(G)
    n = size(G, 1)
    R = zeros(n, n)
    
    for i in 1:n
        for j in 1:n
            if i == j
                R[i, j] = 1.0
            else
                if G[i, i] > 0 && G[j, j] > 0
                    R[i, j] = G[i, j] / sqrt(G[i, i] * G[j, j])
                else
                    R[i, j] = 0.0
                end
            end
        end
    end
    return R
end

# Calculate genetic correlation matrix
R = cov_to_corr(G)

println("Genetic Correlation Matrix:")
println("=" ^ 40)
R_df = DataFrame(R, trait_names)
insertcols!(R_df, 1, :Trait => trait_names)
println(R_df)
println()

# Extract unique correlations and create results table
correlation_results = DataFrame(
    Trait1 = String[],
    Trait2 = String[],
    Genetic_Correlation = Float64[],
    Covariance = Float64[],
    Trait1_Variance = Float64[],
    Trait2_Variance = Float64[]
)

for i in 1:n_traits
    for j in (i+1):n_traits
        push!(correlation_results, [
            trait_names[i],
            trait_names[j],
            R[i, j],
            G[i, j],
            G[i, i],
            G[j, j]
        ])
    end
end

# Calculate standard errors using Delta method approximation
# For correlation r_ij = σ_ij / sqrt(σ_ii * σ_jj)
# SE approximation requires the standard errors of the variance components

function calculate_correlation_se(cov_ij, var_i, var_j, se_cov_ij, se_var_i, se_var_j)
    # Delta method for correlation standard error
    # This is an approximation - exact calculation would require full covariance matrix of estimates
    
    if var_i <= 0 || var_j <= 0
        return 0.0
    end
    
    r_ij = cov_ij / sqrt(var_i * var_j)
    
    # Partial derivatives
    d_r_d_cov = 1 / sqrt(var_i * var_j)
    d_r_d_var_i = -0.5 * cov_ij / (var_i * sqrt(var_i * var_j))
    d_r_d_var_j = -0.5 * cov_ij / (var_j * sqrt(var_i * var_j))
    
    # Approximate standard error (assuming independence of estimates)
    se_r = sqrt(
        (d_r_d_cov * se_cov_ij)^2 +
        (d_r_d_var_i * se_var_i)^2 +
        (d_r_d_var_j * se_var_j)^2
    )
    
    return se_r
end

# Add standard errors to results (initialize with correct length)
correlation_results[!, :SE_Correlation] = zeros(Float64, nrow(correlation_results))

# Create lookup for standard errors
se_dict = Dict()
for row in eachrow(genetic_var_df)
    cov_name = string(row.Covariance)
    se_dict[cov_name] = row.SD
end

for i in 1:nrow(correlation_results)
    trait1 = correlation_results[i, :Trait1]
    trait2 = correlation_results[i, :Trait2]
    
    # Get standard errors
    se_cov = get(se_dict, "$(trait1)_$(trait2)", get(se_dict, "$(trait2)_$(trait1)", 0.0))
    se_var1 = get(se_dict, "$(trait1)_$(trait1)", 0.0)
    se_var2 = get(se_dict, "$(trait2)_$(trait2)", 0.0)
    
    # Calculate correlation SE
    se_corr = calculate_correlation_se(
        correlation_results[i, :Covariance],
        correlation_results[i, :Trait1_Variance],
        correlation_results[i, :Trait2_Variance],
        se_cov, se_var1, se_var2
    )
    
    # Store the calculated SE
    correlation_results[i, :SE_Correlation] = se_corr
end

# Add confidence intervals (approximate)
correlation_results[!, :CI_Lower] = correlation_results[!, :Genetic_Correlation] - 1.96 * correlation_results[!, :SE_Correlation]
correlation_results[!, :CI_Upper] = correlation_results[!, :Genetic_Correlation] + 1.96 * correlation_results[!, :SE_Correlation]

# Display results
println("Genetic Correlations with Standard Errors:")
println("=" ^ 60)
for i in 1:nrow(correlation_results)
    row = correlation_results[i, :]
    println("$(row.Trait1) - $(row.Trait2):")
    println("  Correlation: $(round(row.Genetic_Correlation, digits=4)) ± $(round(row.SE_Correlation, digits=4))")
    println("  95% CI: [$(round(row.CI_Lower, digits=4)), $(round(row.CI_Upper, digits=4))]")
    println()
end

# Clean up results for final display
final_results = select(correlation_results, 
    :Trait1, :Trait2, :Genetic_Correlation, :SE_Correlation, :CI_Lower, :CI_Upper)

println("Summary Table:")
println(final_results)

# Save results
output_path = "C:\\Users\\joah\\OneDrive - Skogforsk\\Documents\\Projekt\\Optimum contribution selection\\NorwaySpruceData\\Save\\genetic_correlations_results_H_5525.csv"
CSV.write(output_path, final_results)
println("\nResults saved to 'genetic_correlations_results_H_5525.csv'")

# Create visualizations
println("\nCreating visualizations...")

try
    using Plots, StatsPlots
    
    # Set plot defaults (using valid parameters)
    gr()  # Use GR backend
    
    # 1. Heatmap of genetic correlation matrix
    p1 = heatmap(R, 
                xticks=(1:n_traits, trait_names),
                yticks=(1:n_traits, trait_names),
                title="Genetic Correlation Matrix",
                color=:bluesreds,
                clims=(-1, 1),
                aspect_ratio=:equal,
                size=(500, 400),
                titlefontsize=10)
    
    # Add correlation values as text on heatmap
    for i in 1:n_traits
        for j in 1:n_traits
            annotate!(j, i, text(string(round(R[i,j], digits=2)), 
                     R[i,j] > 0.5 || R[i,j] < -0.5 ? :white : :black, 8))
        end
    end
    
    # 2. Bar plot with error bars for genetic correlations
    trait_pairs = [final_results[i, :Trait1] * "-" * final_results[i, :Trait2] for i in 1:nrow(final_results)]
    correlations = final_results[!, :Genetic_Correlation]
    se_values = final_results[!, :SE_Correlation]
    
    p2 = bar(trait_pairs, correlations,
            yerror=se_values,
            title="Genetic Correlations with Standard Errors",
            xlabel="Trait Pairs",
            ylabel="Genetic Correlation",
            xrotation=45,
            legend=false,
            color=ifelse.(correlations .>= 0, :steelblue, :coral),
            size=(800, 500),
            titlefontsize=10)
    
    # Add horizontal line at zero
    hline!([0], color=:black, linestyle=:dash, alpha=0.5)
    
    # 3. Confidence interval plot
    p3 = scatter(1:nrow(final_results), correlations,
                yerror=(correlations .- final_results[!, :CI_Lower], 
                       final_results[!, :CI_Upper] .- correlations),
                title="Genetic Correlations with 95% Confidence Intervals",
                xlabel="Trait Pair",
                ylabel="Genetic Correlation",
                xticks=(1:nrow(final_results), trait_pairs),
                xrotation=45,
                legend=false,
                markersize=6,
                color=:steelblue,
                size=(800, 500),
                titlefontsize=10)
    
    hline!([0], color=:black, linestyle=:dash, alpha=0.5)
    
    # 4. Correlation magnitude plot (absolute values)
    abs_correlations = abs.(correlations)
    sorted_indices = sortperm(abs_correlations, rev=true)
    
    p4 = bar(trait_pairs[sorted_indices], abs_correlations[sorted_indices],
            title="Genetic Correlation Magnitudes (Sorted)",
            xlabel="Trait Pairs",
            ylabel="Absolute Genetic Correlation",
            xrotation=45,
            legend=false,
            color=:lightblue,
            size=(800, 500),
            titlefontsize=10)
    
    # 5. Comparison plot: Correlations vs Standard Errors
    p5 = scatter(abs.(correlations), se_values,
                title="Correlation Magnitude vs Standard Error",
                xlabel="Absolute Genetic Correlation",
                ylabel="Standard Error",
                legend=false,
                markersize=6,
                color=:purple,
                alpha=0.7,
                size=(500, 400),
                titlefontsize=10)
    
    # Add trait pair labels
    for i in 1:length(correlations)
        annotate!(abs(correlations[i]), se_values[i], 
                 text(trait_pairs[i], :bottom, 6, rotation=0))
    end
    
    # 6. Heritability comparison
    herit_df = out["heritability"]
    herit_values = [herit_df[herit_df.Covariance .== trait, :Estimate][1] for trait in trait_names]
    herit_se = [herit_df[herit_df.Covariance .== trait, :SD][1] for trait in trait_names]
    
    p6 = bar(trait_names, herit_values,
            yerror=herit_se,
            title="Heritabilities with Standard Errors",
            xlabel="Traits",
            ylabel="Heritability",
            legend=false,
            color=:lightgreen,
            size=(600, 400),
            titlefontsize=10)
    
    # Combine plots in a layout
    layout = @layout [a b; c d; e f]
    combined_plot = plot(p1, p2, p3, p4, p5, p6, layout=layout, size=(1200, 1200))
    
    # Save individual plots
    plot_path = "C:\\Users\\joah\\OneDrive - Skogforsk\\Documents\\Projekt\\Optimum contribution selection\\NorwaySpruceData\\Figures\\"
    suffix = "H_5525"
    savefig(p1, plot_path * "genetic_correlation_heatmap_" * suffix * ".png")
    savefig(p2, plot_path * "genetic_correlation_barplot_" * suffix * ".png")
    savefig(p3, plot_path * "genetic_correlation_ci_" * suffix * ".png")
    savefig(p4, plot_path * "genetic_correlation_magnitude_" * suffix * ".png")
    savefig(p5, plot_path * "correlation_vs_se_" * suffix * ".png")
    savefig(p6, plot_path * "heritabilities_" * suffix * ".png")
    savefig(combined_plot, plot_path * "genetic_correlations_combined_" * suffix * ".png")
    savefig(p1, plot_path * "genetic_correlation_heatmap_" * suffix * ".pdf")
    savefig(p2, plot_path * "genetic_correlation_barplot_" * suffix * ".pdf")
    savefig(p3, plot_path * "genetic_correlation_ci_" * suffix * ".pdf")
    savefig(p4, plot_path * "genetic_correlation_magnitude_" * suffix * ".pdf")
    savefig(p5, plot_path * "correlation_vs_se_" * suffix * ".pdf")
    savefig(p6, plot_path * "heritabilities_" * suffix * ".pdf")
    savefig(combined_plot, plot_path * "genetic_correlations_combined_" * suffix * ".pdf")
    
    println("Plots saved:")
    println("  - Heatmap: genetic_correlation_heatmap_" * suffix * ".png")
    println("  - Bar plot with SE: genetic_correlation_barplot_" * suffix * ".png") 
    println("  - Confidence intervals: genetic_correlation_ci_" * suffix * ".png")
    println("  - Magnitude plot: genetic_correlation_magnitude_" * suffix * ".png")
    println("  - Correlation vs SE: correlation_vs_se_" * suffix * ".png")
    println("  - Heritabilities: heritabilities_" * suffix * ".png")
    println("  - Combined plot: genetic_correlations_combined_" * suffix * ".png")
    
    # Display the combined plot
    display(combined_plot)
    
catch LoadError
    println("Plots.jl not available. To create visualizations, install with:")
    println("using Pkg; Pkg.add([\"Plots\", \"StatsPlots\"])")
    
    # Alternative simple text-based visualization
    println("\nText-based correlation matrix:")
    println("=" ^ 50)
    for i in 1:n_traits
        print(rpad(trait_names[i], 10))
        for j in 1:n_traits
            print(rpad(string(round(R[i,j], digits=3)), 8))
        end
        println()
    end
end

# Also display heritabilities for reference
println("\nHeritabilities:")
println("=" ^ 30)
heritability_df = out["heritability"]
println(heritability_df)

# Save results
output_path = "C:\\Users\\joah\\OneDrive - Skogforsk\\Documents\\Projekt\\Optimum contribution selection\\NorwaySpruceData\\Save\\heritability_results_H_5525.csv"
CSV.write(output_path, heritability_df)
println("\nResults saved to 'heritability_results_H_5525.csv'")