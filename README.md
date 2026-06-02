# CP3D — Cell Painting 3D analysis pipeline

End-to-end pipeline for analysing **3D Cell Painting** experiments on HepG2 spheroids (Akura 384). It identifies bioactive compounds that retain activity in 3D, classifies their morphological phenotype, performs cross-plate MoA enrichment, and then validates the findings through several independent downstream analyses: unsupervised multivariate analysis (PCA / t-SNE / UMAP + permutation tests), a 2D-vs-3D comparison against a matched monolayer reference, an orthogonal deep-learning representation (DINOv2), chemical-diversity / chemotype analysis, MoA-recall benchmarks, and external validation against DILI and LINCS L1000.

> **Project context**: WP3 of EU-OPENSCREEN. The ECBL library (735 compounds preselected from ~2,500 hits in a 2D screen by Wolff et al., *iScience* 2025) tested at 10 µM on HepG2 spheroids. Cell Painting with 4 fluorophores, Operetta CLS, 5 Z-planes, 4 channels (ER, AGP, Mito, DNA), maximum-intensity projection (MIP).

---

## 🧭 How the repository is organised

The codebase has **two tiers**:

1. **Core pipeline — stages 1–9 (orchestrated).** Raw CellProfiler outputs → hits → cross-plate enrichment → multivariate validation. This is fully automated by `run_full_pipeline.py` (single command, ~2–3 min) and is the part that produces the 25 robust hits.

2. **Downstream / paper-specific analyses — scripts 10–21 plus two extras (standalone).** Each is run individually and consumes the core outputs (mainly `data/processed/consensus_*.csv` and `data/annotated/cp3d_library_annotated.csv`). These build the figures and statistics for the manuscript: 2D-vs-3D orthogonality, DINOv2 validation, chemical diversity / Tanimoto, the HSP90 chemotype case study, MoA recall, DILI prediction and LINCS L1000 transcriptomic validation.

> Numbering note: `10_build_annotated_library.py` is the bridge between the two tiers — it must run after the core pipeline (it needs the consensus files and hit list) and before any of the downstream scripts (11–21), which all read `data/annotated/cp3d_library_annotated.csv`.

---

## 📋 Quick start

```bash
# Activate conda environment
conda activate cp3d

# 1) CORE PIPELINE (stages 1-9) — cleans previous outputs and reprocesses everything
cd C:\Users\Ianezi\Documents\CP3D\analysis\scripts
python run_full_pipeline.py

# Process specific plates only
python run_full_pipeline.py --plates C2386 C2388

# Skip cleanup (keep existing outputs and re-run)
python run_full_pipeline.py --skip_clean

# Re-run only the multivariate stage with a custom target group
python run_full_pipeline.py --start_from 9 ^
    --target_group EOS101430 EOS101302 ^
    --target_label expanded

# 2) BRIDGE — build the annotated library consumed by all downstream scripts
python 10_build_annotated_library.py

# 3) DOWNSTREAM (run individually, after step 2)
python 11_compare_2d_medina_vs_3d.py
python chemical_diversity_25hits.py
python 15_dinov2_embeddings.py        # requires raw MIP images + torch (slow, GPU recommended)
python 16_dinov2_analysis.py
# ... etc (see "Downstream analyses" table below)
```

Total time for the core pipeline: **~2–3 minutes** for 3 plates, 12 replicates and 735 unique compounds (plus ~30 s for the multivariate stage). Downstream scripts vary: most run in seconds–minutes; `15_dinov2_embeddings.py` is the exception (depends on image I/O and GPU).

---

## 🗂️ Project structure

```
C:\Users\Ianezi\Documents\CP3D\
│
├── 📄 EOS_compounds_smiles.csv             # Library: SMILES + IDs
├── 📄 EOS_compounds_MoA.csv                # Library: MoA + targets + ChEMBL/GTOPDB
├── 📄 EOS_compounds_MoA_HITs.csv           # Annotated subset of hits
│
├── 📁 Platemaps/                           # Original Excel plate maps + ECHO protocol
├── 📁 paper_draft/                         # Manuscript draft + vector figures (not used by code)
│
└── 📁 analysis/
    ├── 📄 README.md
    ├── 📄 Paper_v9_edited.docx
    │
    ├── 📁 scripts/                         # Pipeline + downstream scripts
    │   │  ── core, orchestrated (stages 1-9) ──
    │   ├── 🔧 process_platemaps.py
    │   ├── 🔧 process_platemap_C2388.py    # multi-replicate plate special case
    │   ├── 🔧 split_C2388_replicates.py    # splits C2388 into 4 virtual folders
    │   ├── 🔧 01_merge_and_qc.py
    │   ├── 🔧 02_normalize.py
    │   ├── 🔧 03_combine_and_score.py
    │   ├── 🔧 04_robust_analysis.py
    │   ├── 🔧 05_merge_hits_chemistry.py
    │   ├── 🔧 06_compare_hits_vs_library.py
    │   ├── 🔧 07_combined_analysis.py
    │   ├── 🔧 08_visualize_all_hits.py
    │   ├── 🔧 09_multivariate_analysis.py  # PCA + t-SNE + UMAP + permutation tests
    │   ├── 🚀 run_full_pipeline.py         # master script (orchestrates stages 1-9)
    │   │
    │   │  ── bridge ──
    │   ├── 🔧 10_build_annotated_library.py
    │   │
    │   │  ── downstream / paper analyses (standalone) ──
    │   ├── 🔧 11_compare_2d_medina_vs_3d.py
    │   ├── 🔧 12_target_signals_2d_vs_3d.py
    │   ├── 🔧 13_figure_signal_complementarity.py
    │   ├── 🔧 14_dili_predictor.py
    │   ├── 🔧 14a_enrich_dili_pubchem.py
    │   ├── 🔧 15_dinov2_embeddings.py
    │   ├── 🔧 16_dinov2_analysis.py
    │   ├── 🔧 17_moa_recall.py
    │   ├── 🔧 18_moa_recall_curated.py
    │   ├── 🔧 19_moa_recall_repurposing_hub.py
    │   ├── 🔧 20_hsp90_chemotype_figure.py
    │   ├── 🔧 21_lincs_l1000_validation.py  (+ 21a–21d API exploration helpers)
    │   ├── 🔧 chemical_diversity_25hits.py  # Tanimoto + scaffolds (Supp Fig S8)
    │   └── 🔧 compare_hsp90_stk24.py        # 6-compound side-by-side
    │
    ├── 📁 data/
    │   ├── 📁 raw/                          # CellProfiler outputs (do NOT modify)
    │   │   ├── C2386R1..R4/, C2387R1..R4/
    │   │   └── C2388/  + C2388R1..R4/       # split into virtual folders
    │   ├── 📁 platemaps/processed/          # generated, regeneratable
    │   ├── 📁 processed/                    # generated: merged / normalized / consensus CSVs
    │   ├── 📁 annotated/                    # cp3d_library_annotated.csv (+ audit)
    │   ├── 📁 embeddings/                   # DINOv2 1536-dim embeddings (parquet)
    │   └── 📁 external/                     # reference datasets (see below)
    │
    └── 📁 results/                          # all generated, regeneratable
        ├── 📁 qc/                           # QC plots per replicate
        ├── 📁 norm/                         # normalization QC
        ├── 📁 multi_rep/                    # multi-replicate analysis
        ├── 📁 hits_summary/                 # hits + chemistry/MoA
        ├── 📁 enrichment/  └ combined/      # per-plate + cross-plate enrichment
        ├── 📁 integrated_hits/              # integrated hit visualisations
        ├── 📁 multivariate/  └ HSP90/, STK24/   # PCA/t-SNE/UMAP + proximity tests per target group
        ├── 📁 medina_2d_vs_3d/              # 2D-vs-3D comparison (scripts 11-13)
        ├── 📁 chemical_diversity/           # Tanimoto / scaffolds (25 hits)
        ├── 📁 hsp90_chemotype/              # HSP90 chemotype panels (script 20)
        ├── 📁 hsp90_vs_stk24/               # 6-compound comparison
        ├── 📁 dinov2_analysis/              # DINOv2 validation (scripts 15-16)
        ├── 📁 moa_recall/, moa_recall_curated/, moa_recall_repurposing_hub/
        ├── 📁 dili/                         # DILI prediction (scripts 14, 14a)
        └── 📁 lincs_l1000/                  # transcriptomic validation (script 21)
```

### External reference datasets (`data/external/`)

| File | Source | Used by |
|------|--------|---------|
| `MEDINA_HepG2_norm_reduced_filtered_median.csv` (+ `_active_mask.csv`) | Wolff et al., *iScience* 28, 112445 (2025); Zenodo 10.5281/zenodo.13309566 | 11–13, 14, 20 |
| `wolff_2025_per_site/` | Wolff et al. 2025, per-site profiles | 11 |
| `DILIrank.xlsx` (+ `DILIrank_pubchem_enriched.csv`) | FDA LTKB DILIrank (Chen et al., *Drug Discov Today* 2016) | 14, 14a |
| `repurposing_drugs.txt`, `repurposing_samples.txt` | Broad Drug Repurposing Hub (Corsello et al., *Nat Med* 2017) | 19 |

---

## 🔬 Core pipeline (stages 1–9)

| Stage | Script(s) | Function | Key outputs |
|-------|-----------|----------|-------------|
| **1** | `process_platemaps.py`, `process_platemap_C2388.py` | Process plate maps. C2388 special case: 4 technical replicates in one physical plate, dispensed in column blocks (R1=cols 1–5, R2=6–10, R3=11–15, R4=16–20). | `data/platemaps/processed/platemap_*_processed.csv` |
| **2** | `split_C2388_replicates.py` | Split C2388 raw CSVs into 4 virtual folders (C2388R1–R4) so the standard pipeline can process them. | `data/raw/C2388R1..R4/` |
| **3** | `01_merge_and_qc.py`, `02_normalize.py` | Merge CellProfiler features with plate map + QC plots. DMSO-based normalization (`mad_robustize`, MAD floor = 0.01, winsorize ±10), then feature selection via pycytominer (variance, correlation, NA). | `*_merged.csv`, `*_normalized_dmso.csv`, `*_normalized_dmso_selected.csv` |
| **4** | `03_combine_and_score.py`, `04_robust_analysis.py` | Combine replicates into a per-compound consensus, compute activity score (RMS / Euclidean distance from DMSO), and apply the first two hit-calling criteria (activity > P95 AND replicate correlation > 0.3, the latter computed only on active candidates; the third criterion, N ≥ 3 valid replicates, is applied at stage 5). HDBSCAN clustering of confirmed hits. | `confirmed_hits_<plate>.csv` ⭐, `consensus_<plate>.csv`, `global_combined_<plate>.csv` |
| **5** | `05_merge_hits_chemistry.py` | Cross hits with `EOS_compounds_MoA.csv` (drug name, target, gene, MoA, n_targets, ChEMBL/GTOPDB hierarchy, SMILES). Apply `--min_replicates` filter (default 3) to separate robust hits from those excluded for low N (audit trail). | `<plate>_hits_with_chemistry.csv`, `<plate>_hits_excluded_low_reps.csv` |
| **6** | `06_compare_hits_vs_library.py` | Per-plate Fisher exact enrichment (3D hits vs lost-in-3D) for MoA, target, gene and ChEMBL/GTOPDB classification (4 levels), with Benjamini-Hochberg FDR. | `results/enrichment/` |
| **7** | `07_combined_analysis.py` | Cross-plate analysis (735 compounds = 25 robust hits + 710 lost). Promiscuity analysis (n_targets) hits vs lost: Mann-Whitney + Fisher exact, with per-plate breakdown. | `results/enrichment/combined/` |
| **8** | `08_visualize_all_hits.py` | 6 integrated plots: master scatter (Activity × Area), reproducibility, phenotype distribution, summary table, cross-plate target convergence, promiscuity bubble plot. | `results/integrated_hits/` |
| **9** | `09_multivariate_analysis.py` | Unsupervised PCA + t-SNE + UMAP on 735 compounds × **112 common features** (intersection of features surviving selection in all 3 plates). Permutation tests of target-group cohesion; phenotype validation via Mann-Whitney distances + silhouette. Per-target-group output folders (`HSP90/`, `STK24/`). | `results/multivariate/<label>/` |

**Stage 9 tests:** pairwise-distance percentile of the target group within all hit pairs · permutation test (10,000 iters) vs random hit triplets and vs random library triplets · one-sided Mann-Whitney on intra- vs inter-phenotype distances · silhouette score observed vs label-shuffle null (1,000 perms).

---

## 🧬 Downstream analyses (scripts 10–21 + extras)

Run individually after the core pipeline and `10_build_annotated_library.py`.

| Script | Purpose | Output folder |
|--------|---------|---------------|
| `10_build_annotated_library.py` | **Bridge.** Build the single annotated table of 735 compounds (SMILES, InChIKey, target, MoA, `is_hit_3D`) consumed by all downstream scripts. | `data/annotated/` |
| `11_compare_2d_medina_vs_3d.py` | 2D-vs-3D orthogonality vs matched Wolff 2025 HepG2 monolayer profiles (same cell line, site, library, modality; only architecture differs). Spearman ρ, Mann-Whitney, per-compound rank percentiles. | `results/medina_2d_vs_3d/` |
| `12_target_signals_2d_vs_3d.py` | Targets/MoAs enriched in the top-25 2D vs top-25 3D compounds (matched N), Fisher exact. | `results/medina_2d_vs_3d/target_signals/` |
| `13_figure_signal_complementarity.py` | Publication figure: Venn of top-25 sets + mirror enrichment plot (2D vs 3D). | `.../target_signals/figures/` |
| `14_dili_predictor.py` | Tests whether 2D/3D morphological activity predicts FDA DILIrank clinical liver-injury class. ROC/AUC with bootstrap CI, Fisher exact. | `results/dili/` |
| `14a_enrich_dili_pubchem.py` | One-off helper: enrich DILIrank with PubChem InChIKeys (cached) for reliable matching. | `data/external/DILIrank_pubchem_enriched.csv` |
| `15_dinov2_embeddings.py` | Compute DINOv2 ViT-S/14 embeddings from raw MIP images, per channel (4 × 384 = **1536-dim** per compound). Self-supervised, fully independent of CellProfiler. *Requires raw images + torch.* | `data/embeddings/` |
| `16_dinov2_analysis.py` | Validate phenotypic clusters in DINOv2 space: HSP90 permutation test, Mantel test vs CellProfiler distances, per-channel (ER/AGP/MITO/DNA) contribution. | `results/dinov2_analysis/` |
| `17_moa_recall.py` | MoA-recall benchmark (AUC + mAP@K) in CellProfiler vs DINOv2 spaces, using raw ChEMBL target annotation. | `results/moa_recall/` |
| `18_moa_recall_curated.py` | Same benchmark with ~15 manually curated target families (cleaner ground truth than raw ChEMBL). | `results/moa_recall_curated/` |
| `19_moa_recall_repurposing_hub.py` | Same benchmark using Broad Drug Repurposing Hub primary-mechanism annotation. | `results/moa_recall_repurposing_hub/` |
| `20_hsp90_chemotype_figure.py` | 4 publication panels: 2D-vs-3D scatter, activity by chemotype, distance heatmap, rank bars — the ansamycin/tanespimicina chemotype story. | `results/hsp90_chemotype/` |
| `21_lincs_l1000_validation.py` | Transcriptomic orthogonal validation: HSF1 heat-shock-response signature of HSP90 compounds via SigCom LINCS L1000. (`21a–21d` are API-exploration helpers.) | `results/lincs_l1000/` |
| `chemical_diversity_25hits.py` | Chemical (Bemis-Murcko scaffolds + pairwise Tanimoto, Morgan r=2, 2048-bit) vs phenotypic diversity of the 25 hits — identifies the HSP90 ansamycins as the only chemotype-coherent group (Supp Fig S8). | `results/chemical_diversity/` |
| `compare_hsp90_stk24.py` | Side-by-side comparison of the HSP90 triplet vs the STK24 triplet: 6×6 Pearson correlation, top discriminating features, multi-organelle granularity signature. | `results/hsp90_vs_stk24/` |

---

## 🧪 Methodology — key decisions

**Image acquisition / processing**
- **MIP 2D** instead of full 3D (reduces the dataset ~5×, suitable for Akura geometry).
- **Whole spheroid** segmented as a single object (nuclei overlap in the MIP).
- **No `CorrectIlluminationCalculate`** (Akura plate geometry confounds it).

**Failed wells (~9%)** — NOT treated as biological toxicity; consistent with Echo dispensation artefacts (dispersed, random spatial pattern). Excluded from analysis but reported in QC.

**Normalization** — DMSO-based (not against active compounds, since ECBL is bioactive). MAD floor 0.01 + winsorize ±10 defend against outliers. Per-plate, so batch effects between plates are corrected automatically.

**Hit calling (three-criterion)** — activity score (RMS) > P95 of the plate **AND** replicate correlation > 0.3 (computed only on active candidates) **AND** ≥ 3 valid replicates. Hits with N < 3 are documented in `<plate>_hits_excluded_low_reps.csv` for transparency.

**Phenotype classification** (consensus `AreaShape_Area` z-score) — `cytotoxic_compact` / shrunken (z < −1), `disgregating` / expanded (z > +1), `morphological_other` / stable (|z| < 1).

**C2388 special case** — 4 technical replicates in one physical plate (column blocks of 5); detected automatically via the `Replicate` column. Permissive thresholds (P80/P70) compensate for the small library (35 compounds).

**Multivariate space** — intersection of features surviving selection independently in each of the 3 plates (**112 common features**); per-feature z-standardization on the 735-compound matrix; Euclidean distance; Ward linkage for heatmaps. All tests are non-parametric / permutation-based.

---

## 📊 Results summary

### Plate-level results

| Plate | Compounds | Robust hits (N≥3) | Excluded (N<3) | Phenotypes |
|-------|-----------|-------------------|----------------|------------|
| **C2386** | 349 | 9 | 6 | 6 shrunken, 2 expanded, 1 stable |
| **C2387** | 352 | 14 | 4 | 6 shrunken, 8 stable |
| **C2388** | 35 | 2 | 0 | 1 shrunken, 1 stable |
| **TOTAL** | **735** | **25** | **10** | **13 / 2 / 10** |

### Enrichment (735 compounds, 25 hits vs 710 lost)

- **Combined level:** 0 enrichments significant at FDR < 0.25 (largest signals: ChEMBL Level 2 "Peptidases and proteinases" FDR = 0.16; "Heat shock proteins" FDR = 0.65). The robust HSP90 signal in C2387 dilutes when all plates are combined.
- **C2387 individual analysis — 4 enrichments at FDR < 0.05:** HSP90B1 (FDR = 0.038), HSP90AB1 (FDR = 0.038) — both the 3 ansamycins; CDK3 (FDR = 0.029) and MAP4K5 (FDR = 0.042), both driven by highly promiscuous compounds (interpret with caution).
- **Promiscuity (n_targets):** hits median 9 vs lost 6; Mann-Whitney p = 0.36 globally (NOT significant) — surviving in 3D does not generally require promiscuity. Plate-dependent: significant only in C2387 (10.5 vs 4 targets, p = 0.0039).

### Multivariate validation (stage 9 — CellProfiler space)

- PCA, t-SNE and UMAP all place the 3 HSP90 inhibitors as close neighbours.
- Pair distances at percentiles **4.7, 15.7 and 19.0** of all hit-pair distances.
- **Permutation test vs random hit triplets: p = 0.043** ⭐ (significant). Mean triplet distance 17.31.
- Permutation test vs random library triplets: p = 0.84 (signal is local, not global).
- STK24 triplet: p = 0.075 (does not reach significance) — kept in limitations, not as a case study.
- **Phenotype classification:** silhouette = 0.032 vs random-label median −0.055, permutation p = 0.004 ⭐. `shrunken` clusters strongly (p < 0.0001), `stable` significantly (p = 0.018), `expanded` (n = 2) underpowered (p = 0.53).

### DINOv2 orthogonal validation (scripts 15–16)

- HSP90 ansamycin cluster in full 1536-dim DINOv2 space: **p = 0.068** (trend; CellProfiler reference p = 0.043). Reproduced with lower power, not destroyed → not a feature-engineering artefact.
- **Mantel test CellProfiler vs DINOv2** (25 hits): Spearman ρ = 0.628, Pearson r = 0.608, permutation p < 10⁻⁴ — the two spaces agree on how the hits group.
- Per-channel HSP90 cluster strength: **AGP p = 0.067** (most discriminant) > DNA p = 0.087 > ER p = 0.092 > MITO p = 0.176. Cytoskeleton + nucleus carry the signal — consistent with ROS-mediated damage by the ansamycin benzoquinone.

### Chemical diversity (chemical_diversity_25hits.py)

Of the 300 possible pairs among the 25 hits, only **3 exceed Tanimoto 0.30** — and all 3 are the ansamycin pairs (geldanamycin/alvespimycin 0.81, geldanamycin/retaspimycin 0.80, alvespimycin/retaspimycin 0.76). The HSP90 ansamycins are the only chemotype-coherent group in the hit set.

### Caveat — the HSP90 cluster is not exclusive to HSP90
The HSP90 inhibitors share their phenotypic neighbourhood with DELANZOMIB (proteasome), DEQUALINIUM (mitochondrial), AZT, BENZO[A]PYRENE, A-196 and several kinase inhibitors. The Cell Painting phenotype alone is therefore **not diagnostic** of HSP90 inhibition — biochemical validation (CETSA, client Western blot) remains essential.

### Validated hits (clinical/research relevance)

- **C2387 (main finding):** GELDANAMYCIN, RETASPIMYCIN, ALVESPIMYCIN (3 HSP90 ansamycins) ⭐⭐; DELANZOMIB (proteasome); OTS964 (TOPK/PBK, very selective); DOVITINIB, XL228, ZOTIRACICLIB (promiscuous kinase inhibitors); ABT-702, F16, G-5555, ML351, ELLIPTICINE, PD027679.
- **C2386:** OMACETAXINE (RPL3, AML — expanded, very selective) ⭐; ENTOSPLETINIB (SYK/BTK — expanded); DEQUALINIUM; BENZO[A]PYRENE (carcinogen control); AZT; ETP-46464, A-196, LOSMAPIMOD, METERGOLINE.
- **C2388:** COUMARIN 7 (⚠ possible fluorescent artefact — verify channel cross-talk); PD129966.

---

## 🛠️ Dependencies

```bash
conda create -n cp3d python=3.11
conda activate cp3d

# core pipeline
pip install pandas numpy matplotlib scipy scikit-learn
pip install pycytominer
pip install hdbscan umap-learn        # clustering (stage 4) + UMAP (stage 9)
pip install openpyxl                  # Excel plate maps

# downstream
pip install rdkit                     # scaffolds + Tanimoto (chemical_diversity)
pip install pubchempy                 # DILI InChIKey enrichment (14a)
pip install torch torchvision         # DINOv2 embeddings (15) — GPU recommended
pip install requests                  # LINCS / SigCom API (21)
```

**Notes:** `umap-learn` is optional for stage 9 (falls back to PCA + t-SNE if missing). `15_dinov2_embeddings.py` needs the raw MIP TIFFs and is the only computationally heavy step; once `data/embeddings/*.parquet` exist, `16`–`19` run from the cached embeddings without re-imaging.

---

## 🔁 Reproducibility

```bash
# Reprocess the core pipeline from raw CSVs
python run_full_pipeline.py

# Verify outputs
dir data\processed              # consensus + normalized CSVs
dir results\integrated_hits     # 6 PNG + 1 CSV
dir results\multivariate\HSP90  # PCA/t-SNE/UMAP + proximity tests + summary
```

The core pipeline is deterministic given the same raw data and `--random_state` (default 42); identical numbers between runs. t-SNE and UMAP have stochastic components fixed via the seed. Downstream scripts that hit live web APIs (PubChem in 14a, SigCom LINCS in 21) depend on those services being reachable and may change as the external databases are updated; their fetched data is cached locally where possible.

---

## ⚠️ Limitations and considerations

1. **n_hits is low (25)** for FDR-corrected enrichment o