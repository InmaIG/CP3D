# CP3D — Cell Painting 3D analysis pipeline

End-to-end pipeline for **3D Cell Painting** in HepG2 spheroids (Akura 384). It turns confocal Z-stacks into per-compound morphological profiles, calls robust hits, classifies their phenotype, and validates the result through several independent analyses (a paired 2D-vs-3D comparison, a self-supervised DINOv2 embedding, chemical-diversity / chemotype analysis, and MoA-recall benchmarks).

> **Context** — WP3 of EU-OPENSCREEN. 735 bioactive compounds (active in the 2D HepG2 screen of Wolff et al., *iScience* 2025) tested at 10 µM; Cell Painting (4 channels: ER, AGP, Mito, DNA) on an Operetta CLS; maximum-intensity projection (MIP); CellProfiler features. The pipeline recovers **25 robust 3D hits** and resolves chemotype-specific bioactivity within the HSP90 inhibitor class.

## Repository layout — two tiers

1. **Core pipeline (stages 1–9), orchestrated.** `run_full_pipeline.py` runs the whole chain (raw features → normalisation → hit calling → cross-plate enrichment → multivariate validation) in ~2–3 min and produces the 25 robust hits.
2. **Downstream analyses (scripts 10–21 + extras), standalone.** Run individually; each consumes the core outputs and builds a manuscript figure or statistic. `10_build_annotated_library.py` is the bridge — run it after the core pipeline and before 11–21.

## Quick start

```bash
conda activate cp3d
cd analysis/scripts

# Core pipeline (stages 1-9)
python run_full_pipeline.py                  # all plates
python run_full_pipeline.py --plates C2386   # a specific plate
python run_full_pipeline.py --start_from 9 --target_group EOS101988 EOS100193 EOS100198 --target_label HSP90

# Bridge, then downstream (examples)
python 10_build_annotated_library.py
python 11_compare_2d_medina_vs_3d.py
python chemical_diversity_25hits.py
```

## Structure

```
CP3D/
├── EOS_compounds_*.csv           # library: SMILES, MoA, targets
├── analysis/
│   ├── scripts/                  # pipeline 01-09 + bridge 10 + downstream 11-21 (+ extras)
│   ├── data/    (git-ignored)    # raw/, processed/, annotated/, embeddings/, external/
│   └── results/ (git-ignored)    # qc/, multivariate/, dinov2_analysis/, chemical_diversity/, ...
└── Platemaps/                    # plate maps + ECHO protocol
```
`data/` and `results/` are regenerable and not tracked in this repository.

## Core pipeline (stages 1-9)

| Stage | Script(s) | Function |
|-------|-----------|----------|
| 1 | `process_platemaps.py`, `process_platemap_C2388.py` | Plate maps (C2388 = 4 replicates in column blocks within one plate). |
| 2 | `split_C2388_replicates.py` | Split C2388 into 4 virtual replicate folders. |
| 3 | `01_merge_and_qc.py`, `02_normalize.py` | Merge features + QC; DMSO `mad_robustize` (MAD floor 0.01, winsorize ±10); pycytominer feature_select. |
| 4 | `03_combine_and_score.py`, `04_robust_analysis.py` | Per-compound consensus; activity score (RMS); first two hit criteria (activity > P95 AND replicate r > 0.3). |
| 5 | `05_merge_hits_chemistry.py` | Add chemistry/MoA; apply the N ≥ 3 filter → robust hits vs low-replicate exclusions. |
| 6 | `06_compare_hits_vs_library.py` | Per-plate Fisher enrichment (Benjamini-Hochberg FDR). |
| 7 | `07_combined_analysis.py` | Cross-plate analysis (25 hits vs 710 lost); promiscuity test. |
| 8 | `08_visualize_all_hits.py` | Integrated hit visualisations. |
| 9 | `09_multivariate_analysis.py` | PCA / t-SNE / UMAP on 735 × 112 features; permutation + silhouette tests. |

**Three-criterion hit calling:** activity_RMS > P95 **and** mean replicate correlation > 0.3 **and** ≥ 3 valid replicates.

## Downstream analyses (run after `10_build_annotated_library.py`)

| Script(s) | Purpose |
|-----------|---------|
| `11`–`13` | 2D-vs-3D orthogonality vs Wolff 2025 (Spearman, Mann-Whitney, complementarity figure). |
| `14`, `14a` | DILI prediction vs FDA DILIrank (ROC/AUC); PubChem InChIKey enrichment. |
| `15`, `16` | DINOv2 embeddings (4 × 384 = 1536-dim) + validation (Mantel vs CellProfiler, per-channel). *15 needs raw images + torch.* |
| `17`–`19` | MoA-recall benchmarks (ChEMBL / curated families / Drug Repurposing Hub). |
| `20` | HSP90 chemotype figure (ansamycins vs tanespimycin). |
| `chemical_diversity_25hits.py` | Tanimoto + Murcko scaffolds of the 25 hits. |
| `compare_hsp90_stk24.py` | HSP90 vs STK24 triplet comparison. |
| `21` (+ `21a`–`21d`) | LINCS L1000 transcriptomic validation (HSF1 heat-shock signature). |

## Key methodological choices

- **MIP** instead of full-volume imaging (≈5× smaller, screening-compatible); **whole spheroid** segmented as a single object.
- **No illumination correction** (the Akura conical-well vignette would confound it); plate bias is corrected at the feature level by per-plate DMSO normalisation.
- **Failed wells (~8%)** treated as Echo-dispensation artefacts (random spatial pattern), not toxicity.
- **Phenotype** from consensus `AreaShape_Area` z-score: shrunken (z < −1), stable (|z| ≤ 1), expanded (z > +1).
- **C2388** (35 compounds) uses permissive thresholds (P80) to compensate for its small size.

## Dependencies

```bash
conda create -n cp3d python=3.11 && conda activate cp3d
pip install pandas numpy scipy scikit-learn matplotlib pycytominer hdbscan umap-learn openpyxl   # core
pip install rdkit pubchempy torch torchvision requests                                           # downstream
```
`umap-learn` is optional (stage 9 falls back to PCA + t-SNE). Only `15_dinov2_embeddings.py` is heavy (needs raw TIFFs + a GPU); later scripts reuse the cached embeddings.

## Reproducibility

The core pipeline is deterministic given the same raw data and `--random_state` (default 42). Scripts that query live APIs (PubChem in `14a`, SigCom LINCS in `21`) depend on those services and cache results locally.

> **Paths:** the scripts currently use an absolute base path. Adjust the `BASE` variable at the top of each script to your local checkout before running.

## Limitations

- Single cell line (HepG2 monoculture); extrapolation to in vivo liver biology needs richer 3D models.
- Thin equatorial Z-stack (10 µm), not full-volume imaging.
- 25 hits is small for combined-level FDR enrichment; categories survive only per plate (C2387).
- COUMARIN 7 (C2388) is a fluorescent dye — its apparent 3D activity may reflect channel cross-talk; retained but flagged.

## Citation & license

Released under the **MIT License** as a community resource accompanying the CP3D manuscript (Iáñez García, Ramos & Fernández-Godino, Fundación MEDINA). If you use this pipeline, please cite the paper and this repository.

**Contact:** Inmaculada Iáñez García — inmaculada.ianez@medinaandalucia.es
