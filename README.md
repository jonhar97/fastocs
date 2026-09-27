# fastOCS

RSVD-OCS: a randomized-SVD approach to Optimum Contribution Selection (OCS) for
large genomic relationship matrices in tree breeding, plus a robust extension
(ROCS) that accounts for breeding-value uncertainty. Methods code for a
manuscript targeting *Bioinformatics* (Ahlinder & Waldmann).

Core idea: OCS requires optimizing over a dense n x n genomic/pedigree
relationship matrix, which is expensive for large breeding populations.
Approximating that matrix with a low-rank factor from a randomized SVD gives
large speedups (targeting ~100x at rank 30-40 for the Norway spruce case
study) with minimal loss in genetic gain or solution quality.

## Repository layout

```
spruce/
  scripts/    Current Norway spruce pipeline (single-trait Hjd17, single-step
              HBLUP via AI-REML)
  data/       Not tracked in git -- see Data below
  output/     Script outputs (CSV/JLD2), not tracked in git
legacy/       Superseded pipeline versions, kept for provenance (see below)
tools/        General-purpose solver not specific to the spruce case study
manuscript/   LaTeX/figures for the paper (not yet populated in this repo)
```

## Current pipeline (spruce/scripts)

Run in this order:

1. **AIREML_spruce.jl** -- single-trait AI-REML (Hjd17), single-step HBLUP.
   Fits `y = Xb + Z_p p + Z_f f + a + e` and writes EBVs, the full PEV matrix,
   and variance components to `spruce/output/`.
2. **load_spruce_hblup.jl** -- shared loader (not run directly; `include()`d
   by the scripts below). Reads the H-matrix and the AI-REML EBVs/PEV into
   `all_ids`, `G`, `g_vec`, `PEV`.
3. **spruce_single_gamma_comparison_hblup.jl** -- full-dense baseline vs.
   PCA-Standard vs. PCA-Randomized (RSVD) OCS across a rank sweep at a fixed
   gamma. Produces the rank-selection and solution-concordance tables/figures.
4. **gamma_rank_sweep_spruce_hblup.jl** -- full-dense vs. RSVD-OCS across a
   gamma x rank grid, with an elbow table recommending a rank per gamma.
5. **robust_ocs_spruce_hblup.jl** -- Robust OCS (ROCS): adds an SQP/cutting-plane
   term penalizing `sqrt(c'*Omega*c)`, where Omega is the AI-REML prediction
   error covariance (PEV) from step 1, represented as a low-rank RSVD factor
   plus an exact diagonal correction. Compares OCS vs. ROCS and benchmarks the
   factorized solve against a full-dense baseline.
6. **make_spruce_figures.py** -- regenerates the manuscript figures from the
   CSV outputs of steps 3-4 (writes into `manuscript/`).

All scripts locate their data/output paths relative to `@__DIR__` (Julia) or
`__file__` (Python), so the layout above works as-is once `spruce/data/`
contains the required input files.

## Data

Phenotype, pedigree/genomic relationship matrix, and EBV/MCMC files are
Skogforsk research data and are **not included** in this repository. Populate
`spruce/data/` locally with (at minimum):
- `Hmat_5525_spruce_tau_1_omega_1_PDF.txt` -- H-matrix (tau=1, omega=1,
  Legarra et al. 2009 formulation), ID column + n x n matrix
- `phenotypes_5525_spruce_Horder_v3.txt` -- phenotypes (Hjd17, Trial,
  Trial_Ruta, Trial_Famly, etc.)

`AIREML_spruce.jl` writes its outputs (EBVs, PEV, variance components) to
`spruce/output/`, which the later scripts then read via `load_spruce_hblup.jl`.

## Legacy (legacy/)

These scripts implement an earlier version of the pipeline and are kept for
provenance rather than active use:

- `JWASGBLUP_5022_spruce.jl` -- original JWAS/MCMC-based multi-trait GBLUP run
  (3-trait index `(Hjd17 + Htv17 - Sprant17)/3`), superseded by the
  single-trait AI-REML/HBLUP approach in `AIREML_spruce.jl`.
- `build_robust_omega.jl` -- built the robust-OCS uncertainty factor (Omega)
  from JWAS MCMC posterior draws. Superseded: the current manuscript uses the
  AI-REML PEV directly as Omega instead (see `robust_ocs_spruce_hblup.jl`),
  to avoid a reviewer question about MCMC-vs-PEV vs. a CVaR framing.
- `gamma_rank_sweep_spruce.jl`, `spruce_single_gamma_comparison.jl`,
  `robust_ocs_spruce_4.jl` -- the corresponding gamma/rank sweep, single-gamma
  comparison, and robust-OCS scripts built around the 3-trait JWAS index and
  MCMC-based Omega. Superseded by the `_hblup` versions in `spruce/scripts/`.

All five legacy scripts use hardcoded local Windows paths
(`C:\Users\JOAH\OneDrive - Skogforsk\...`) from before the pipeline was made
portable -- edit the `BASE_DIR`/`OUT_DIR`/`OMEGA_FACTOR_FILE` constants at the
top of each file before running.

## Tools (tools/)

- `GBLUPAIREML.jl` -- a general-purpose AI-REML GBLUP solver supporting
  missing phenotypes, multiple random effects, and an optional GxE random
  regression model (with LRT/AIC model comparison). Not specific to the
  spruce case study; included as a reusable building block.

## QTL-MAS 2010 benchmark (not yet in this repo)

The manuscript also benchmarks sex-constrained OCS on the QTL-MAS 2010
dataset (n=3,226; RSVD-OCS with separate male/female contribution
constraints, no robust extension). Those scripts
(`qtlmas_ocs_sexconstrained.jl`, `qtlmas_gamma_rank_sweep.jl`,
`qtlmas_gamma_rank_figure.jl`, `qtlmas_compare_solutions_rank30.jl`) did not
transfer in this upload and are not yet in this repository -- add them
whenever convenient.

## Tooling

- Julia: JuMP + OSQP for the OCS QP, JLD2/CSV/DataFrames for I/O,
  LinearAlgebra, StatsBase (Kendall's tau)
- Python: pandas, matplotlib (figure generation only)
- JWAS (legacy pipeline only) for MCMC-based EBV estimation
