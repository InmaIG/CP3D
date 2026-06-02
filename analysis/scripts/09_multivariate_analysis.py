"""
09_multivariate_analysis.py
============================
Multivariate analysis of the CP3D consensus profiles: PCA, t-SNE, UMAP +
permutation tests for proximity / phenotype validation.

Reproduces the four sub-analyses of section 4.7 of the v3 report:

  4.7.1 - Proximity of a target group (default: 3 HSP90 inhibitors) within
          the 25 robust hits, with permutation test against random triplets.
  4.7.2 - Full landscape (735 compounds): PCA, t-SNE, UMAP with hits
          highlighted; permutation test against full library.
  4.7.3 - Phenotype-classification validation (silhouette + Mann-Whitney
          intra vs inter phenotype distances + label-shuffle null).
  4.7.4 - Hierarchical clustering heatmap of the 25 hits and of the top 100
          most active compounds, with hits highlighted.

The script combines the three plates by INTERSECTING their feature sets
(features that survived feature_select within each plate independently).

Usage
-----
    python 09_multivariate_analysis.py
    python 09_multivariate_analysis.py --target_group EOS101988 EOS100193 EOS100198 \
                                       --target_label HSP90
    python 09_multivariate_analysis.py --n_permutations 10000

Inputs
------
    data/processed/consensus_<plate>.csv         (per-plate consensus profile)
    results/hits_summary/<plate>_hits_with_chemistry.csv (robust hits)

Outputs (results/multivariate/)
-------------------------------
    pca_hits_only.png                  Fig. 4.7.1 - 25 hits in PCA + target triangle
    proximity_test_<label>.png         Fig. 4.7.1 - permutation tests
    pca_full_landscape.png             Fig. 4.7.2 - 735 compounds, PCA
    tsne_full_landscape.png            Fig. 4.7.2 - 735 compounds, t-SNE
    umap_full_landscape.png            Fig. 4.7.2 - 735 compounds, UMAP
    phenotype_validation.png           Fig. 4.7.3 - silhouette + Mann-Whitney
    hits_distance_heatmap.png          Fig. 4.7.4 - 25 hits clustered
    top100_compounds_heatmap.png       Fig. 4.7.4 - top 100 with hits highlighted
    multivariate_coordinates.csv       PCA/t-SNE/UMAP coordinates per compound
    proximity_test_results.csv         Permutation test numerical results
    phenotype_validation_stats.csv     Silhouette + Mann-Whitney by phenotype
    nearest_neighbors_<label>.csv      Nearest neighbors of each target compound
    candidates_near_target_group.csv   Top non-hit compounds near target cluster
    multivariate_summary.txt           Narrative text summary
"""

import argparse
import sys
import warnings
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from scipy.stats import mannwhitneyu
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------
BASE = Path(r'C:\Users\Ianezi\Documents\CP3D\analysis')
PROC_DIR = BASE / 'data' / 'processed'
HITS_DIR = BASE / 'results' / 'hits_summary'
# OUT_DIR is defined after argument parsing (depends on TARGET_LABEL)

PLATES = ['C2386', 'C2387', 'C2388']

# Default target group: the 3 HSP90 inhibitors that converged in C2387
DEFAULT_TARGET_GROUP = ['EOS101988', 'EOS100193', 'EOS100198']
DEFAULT_TARGET_LABEL = 'HSP90'
DEFAULT_TARGET_NAMES = {
    'EOS101988': 'GELDANAMYCIN',
    'EOS100193': 'ALVESPIMYCIN',
    'EOS100198': 'RETASPIMYCIN',
}

PLATE_COLORS = {'C2386': '#E74C3C', 'C2387': '#3498DB', 'C2388': '#2ECC71'}
PHENOTYPE_COLORS = {
    'shrunken': '#8E44AD',
    'stable': '#34495E',
    'expanded': '#E67E22',
    'unknown': '#BDC3C7',
}

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--target_group', nargs='+', default=DEFAULT_TARGET_GROUP,
                    help='List of EOS_ids of the target group (default: 3 HSP90 inhibitors)')
parser.add_argument('--target_label', default=DEFAULT_TARGET_LABEL,
                    help='Short label for the target group (used in filenames/plots)')
parser.add_argument('--n_permutations', type=int, default=10000,
                    help='Number of permutations for proximity tests (default: 10000)')
parser.add_argument('--tsne_perplexity', type=float, default=30.0,
                    help='t-SNE perplexity (default: 30; auto-reduced if n<30)')
parser.add_argument('--umap_neighbors', type=int, default=15,
                    help='UMAP n_neighbors (default: 15)')
parser.add_argument('--random_state', type=int, default=42,
                    help='Random seed for reproducibility (default: 42)')
parser.add_argument('--top_n_heatmap', type=int, default=100,
                    help='Top N compounds by activity for top-N heatmap (default: 100)')
args = parser.parse_args()

TARGET_GROUP = list(args.target_group)
TARGET_LABEL = args.target_label
N_PERM = args.n_permutations
RNG = np.random.default_rng(args.random_state)

# Output directory: one subfolder per target group label
OUT_DIR = BASE / 'results' / 'multivariate' / TARGET_LABEL
OUT_DIR.mkdir(parents=True, exist_ok=True)


def log(msg):
    print(msg, flush=True)


def section(title, char='='):
    log("")
    log(char * 70)
    log(f"=== {title} ===")
    log(char * 70)


# ---------------------------------------------------------------------------
# 1. Load and combine plates
# ---------------------------------------------------------------------------
section("STEP 1: Load consensus profiles and intersect features")

per_plate_consensus = {}
for plate in PLATES:
    path = PROC_DIR / f'consensus_{plate}.csv'
    if not path.exists():
        log(f"  ERROR: Missing {path}")
        sys.exit(1)
    df = pd.read_csv(path)
    per_plate_consensus[plate] = df
    log(f"  {plate}: {df.shape[0]} compounds x {df.shape[1]} columns")

# Identify metadata vs feature columns
meta_prefixes = ('Metadata_', 'is_', 'Hit_', 'Phenotype')
all_cols_per_plate = {}
for plate, df in per_plate_consensus.items():
    feat_cols = [c for c in df.columns
                 if not any(c.startswith(p) for p in meta_prefixes)]
    all_cols_per_plate[plate] = set(feat_cols)
    log(f"  {plate}: {len(feat_cols)} feature columns")

# Intersection of feature names across plates
common_features = sorted(set.intersection(*all_cols_per_plate.values()))
log(f"\n  Common features across 3 plates: {len(common_features)}")

if len(common_features) < 20:
    log(f"  ERROR: Too few common features ({len(common_features)}). Cannot proceed.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# 2. Build combined dataframe (one row per compound, common features only)
# ---------------------------------------------------------------------------
section("STEP 2: Build combined compound matrix")

frames = []
for plate, df in per_plate_consensus.items():
    keep_cols = ['Metadata_Compound'] + [c for c in common_features if c in df.columns]
    sub = df[keep_cols].copy()
    sub['Metadata_Plate'] = plate
    # Drop DMSO rows for the multivariate analysis (we want compounds only)
    sub = sub[sub['Metadata_Compound'] != 'DMSO'].reset_index(drop=True)
    frames.append(sub)

combined = pd.concat(frames, ignore_index=True)
combined = combined.drop_duplicates(subset='Metadata_Compound', keep='first').reset_index(drop=True)
log(f"  Combined matrix: {combined.shape[0]} compounds x {len(common_features)} features")

# ---------------------------------------------------------------------------
# 3. Load hits and merge phenotype/drug info
# ---------------------------------------------------------------------------
section("STEP 3: Load hits and merge metadata")

hits_frames = []
for plate in PLATES:
    path = HITS_DIR / f'{plate}_hits_with_chemistry.csv'
    if not path.exists():
        log(f"  WARNING: Missing {path}, skipping")
        continue
    h = pd.read_csv(path)
    hits_frames.append(h[['EOS_id', 'Drug_name', 'Phenotype', 'Activity_score',
                          'Replicate_correlation', 'N_replicates', 'N_targets']]
                       .assign(Plate_origin=plate))

if not hits_frames:
    log("  ERROR: No hits files found.")
    sys.exit(1)

hits = pd.concat(hits_frames, ignore_index=True)
hits = hits.drop_duplicates(subset='EOS_id', keep='first').reset_index(drop=True)
log(f"  Total robust hits: {len(hits)}")

# Annotate combined matrix
combined['is_hit'] = combined['Metadata_Compound'].isin(set(hits['EOS_id']))
combined = combined.merge(
    hits[['EOS_id', 'Drug_name', 'Phenotype', 'Activity_score',
          'Replicate_correlation', 'N_targets']],
    left_on='Metadata_Compound', right_on='EOS_id', how='left'
).drop(columns=['EOS_id'])
combined['Phenotype'] = combined['Phenotype'].fillna('non_hit')

# Verify target group is present
present_targets = [t for t in TARGET_GROUP if t in set(combined['Metadata_Compound'])]
missing_targets = [t for t in TARGET_GROUP if t not in set(combined['Metadata_Compound'])]
log(f"  Target group: {TARGET_LABEL} ({len(TARGET_GROUP)} compounds requested)")
log(f"    Present in dataset: {len(present_targets)} -> {present_targets}")
if missing_targets:
    log(f"    MISSING: {missing_targets}")
    if len(present_targets) < 2:
        log("  Need at least 2 target compounds for proximity analysis. Aborting target tests.")
        TARGET_GROUP = []
    else:
        TARGET_GROUP = present_targets

combined['is_target'] = combined['Metadata_Compound'].isin(set(TARGET_GROUP))

# Compute activity_score for non-hits (raw RMS of consensus profile) so we can rank
profile_matrix_raw = combined[common_features].fillna(0).values
combined['Activity_score_computed'] = np.sqrt((profile_matrix_raw ** 2).mean(axis=1))
# Use reported activity for hits, computed for non-hits
combined['Activity_final'] = combined['Activity_score'].fillna(combined['Activity_score_computed'])

# ---------------------------------------------------------------------------
# 4. Standardize features and compute embeddings
# ---------------------------------------------------------------------------
section("STEP 4: Standardize and embed (PCA, t-SNE, UMAP)")

X = combined[common_features].fillna(0).values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
log(f"  Scaled matrix: {X_scaled.shape}")

# PCA
pca = PCA(n_components=2, random_state=args.random_state)
X_pca = pca.fit_transform(X_scaled)
var_explained = pca.explained_variance_ratio_ * 100
log(f"  PCA done. Variance explained: PC1={var_explained[0]:.1f}%, PC2={var_explained[1]:.1f}%")

# t-SNE
n_total = X_scaled.shape[0]
perplexity = min(args.tsne_perplexity, max(5, (n_total - 1) / 3))
tsne = TSNE(n_components=2, perplexity=perplexity, random_state=args.random_state,
            init='pca', learning_rate='auto')
X_tsne = tsne.fit_transform(X_scaled)
log(f"  t-SNE done (perplexity={perplexity:.0f}).")

# UMAP - try import, fall back gracefully
try:
    import umap
    reducer = umap.UMAP(n_neighbors=args.umap_neighbors, n_components=2,
                        random_state=args.random_state)
    X_umap = reducer.fit_transform(X_scaled)
    log(f"  UMAP done (n_neighbors={args.umap_neighbors}).")
    UMAP_AVAILABLE = True
except ImportError:
    log("  WARNING: umap-learn not installed. Skipping UMAP.")
    log("           Install with: pip install umap-learn")
    X_umap = None
    UMAP_AVAILABLE = False

# Save coordinates
coords = combined[['Metadata_Compound', 'Metadata_Plate', 'Drug_name',
                   'Phenotype', 'is_hit', 'is_target',
                   'Activity_final', 'N_targets']].copy()
coords['PC1'] = X_pca[:, 0]
coords['PC2'] = X_pca[:, 1]
coords['tSNE1'] = X_tsne[:, 0]
coords['tSNE2'] = X_tsne[:, 1]
if UMAP_AVAILABLE:
    coords['UMAP1'] = X_umap[:, 0]
    coords['UMAP2'] = X_umap[:, 1]

coords_path = OUT_DIR / 'multivariate_coordinates.csv'
coords.to_csv(coords_path, index=False)
log(f"  Coordinates saved: {coords_path}")

# Build a hits-only subset
hits_mask = combined['is_hit'].values
hits_idx = np.where(hits_mask)[0]
n_hits = hits_mask.sum()
log(f"  Hits in matrix: {n_hits}")

# Distance matrix (in scaled feature space) for the FULL dataset
log("  Computing pairwise distances (scaled feature space)...")
dist_full = squareform(pdist(X_scaled, metric='euclidean'))
log(f"  Distance matrix shape: {dist_full.shape}")


# ---------------------------------------------------------------------------
# 5. Section 4.7.1 - Proximity of target group within hits
# ---------------------------------------------------------------------------
section(f"STEP 5: Proximity test of target group ({TARGET_LABEL})")

target_idx_in_combined = combined.index[combined['is_target']].tolist()
target_idx_in_hits = [i for i, idx in enumerate(hits_idx) if idx in target_idx_in_combined]

proximity_results = {}

if len(target_idx_in_combined) >= 2 and n_hits >= 5:
    # Pairs within target group
    target_pairs_dist = []
    for i, j in combinations(target_idx_in_combined, 2):
        target_pairs_dist.append(dist_full[i, j])
    target_pairs_dist = np.array(target_pairs_dist)
    target_mean_dist = target_pairs_dist.mean()

    # All pairwise distances among hits (background distribution)
    hit_pair_distances = []
    for i, j in combinations(hits_idx, 2):
        hit_pair_distances.append(dist_full[i, j])
    hit_pair_distances = np.array(hit_pair_distances)

    # Percentile of each target pair within hit pair distribution
    target_percentiles = []
    for d in target_pairs_dist:
        pct = (hit_pair_distances < d).mean() * 100
        target_percentiles.append(pct)

    # Permutation test 1: vs random triplets of hits
    if len(target_idx_in_combined) <= n_hits:
        triplet_size = len(target_idx_in_combined)
        n_perm_real = min(N_PERM, 100000)
        random_means = []
        for _ in range(n_perm_real):
            sample = RNG.choice(hits_idx, size=triplet_size, replace=False)
            d_list = [dist_full[i, j] for i, j in combinations(sample, 2)]
            random_means.append(np.mean(d_list))
        random_means = np.array(random_means)
        p_vs_hits = ((random_means <= target_mean_dist).sum() + 1) / (n_perm_real + 1)
    else:
        random_means = None
        p_vs_hits = np.nan

    # Permutation test 2: vs random triplets of full library
    all_idx = np.arange(combined.shape[0])
    if len(target_idx_in_combined) <= len(all_idx):
        n_perm_real = min(N_PERM, 100000)
        random_means_full = []
        for _ in range(n_perm_real):
            sample = RNG.choice(all_idx, size=triplet_size, replace=False)
            d_list = [dist_full[i, j] for i, j in combinations(sample, 2)]
            random_means_full.append(np.mean(d_list))
        random_means_full = np.array(random_means_full)
        p_vs_full = ((random_means_full <= target_mean_dist).sum() + 1) / (n_perm_real + 1)
    else:
        random_means_full = None
        p_vs_full = np.nan

    proximity_results = {
        'target_label': TARGET_LABEL,
        'target_compounds': ','.join(TARGET_GROUP),
        'n_target': len(target_idx_in_combined),
        'n_hits': n_hits,
        'mean_dist_target': float(target_mean_dist),
        'pair_distances': target_pairs_dist.tolist(),
        'pair_percentiles_within_hits': target_percentiles,
        'median_dist_hits_global': float(np.median(hit_pair_distances)),
        'median_dist_full_library': float(np.median(dist_full[np.triu_indices_from(dist_full, k=1)])),
        'p_vs_random_hit_triplets': float(p_vs_hits) if not np.isnan(p_vs_hits) else None,
        'p_vs_random_library_triplets': float(p_vs_full) if not np.isnan(p_vs_full) else None,
    }

    log(f"  Target mean distance: {target_mean_dist:.3f}")
    log(f"  Median hit-pair distance: {np.median(hit_pair_distances):.3f}")
    log(f"  Pair percentiles within hits: {[f'{p:.1f}' for p in target_percentiles]}")
    if not np.isnan(p_vs_hits):
        log(f"  Permutation p-value (vs random hit triplets): {p_vs_hits:.4f}")
    if not np.isnan(p_vs_full):
        log(f"  Permutation p-value (vs random library triplets): {p_vs_full:.4f}")

    # Save numerical results
    pd.DataFrame([proximity_results]).to_csv(
        OUT_DIR / 'proximity_test_results.csv', index=False)

    # Nearest neighbors of each target compound (within hits)
    nn_rows = []
    for ti in target_idx_in_combined:
        target_eos = combined.iloc[ti]['Metadata_Compound']
        target_drug = combined.iloc[ti]['Drug_name']
        # Distances from this target to all other HITS (excluding self)
        other_hits = [h for h in hits_idx if h != ti]
        dists = [(combined.iloc[h]['Metadata_Compound'],
                  combined.iloc[h]['Drug_name'],
                  dist_full[ti, h]) for h in other_hits]
        dists.sort(key=lambda x: x[2])
        for rank, (eos, drug, d) in enumerate(dists[:5], start=1):
            nn_rows.append({
                'target_eos': target_eos,
                'target_drug': target_drug,
                'rank': rank,
                'neighbor_eos': eos,
                'neighbor_drug': drug,
                'distance': d,
                'is_target_member': eos in TARGET_GROUP,
            })
    pd.DataFrame(nn_rows).to_csv(OUT_DIR / f'nearest_neighbors_{TARGET_LABEL}.csv', index=False)
    log(f"  Nearest-neighbor table saved: nearest_neighbors_{TARGET_LABEL}.csv")
else:
    log(f"  Skipping proximity test (need >=2 target compounds and >=5 hits).")
    random_means = None
    random_means_full = None


# ---------------------------------------------------------------------------
# 6. Section 4.7.2 - Candidates near target cluster (non-hits closest)
# ---------------------------------------------------------------------------
if len(target_idx_in_combined) >= 2:
    section("STEP 6: Identify non-hit compounds near target cluster")
    non_hit_idx = combined.index[~combined['is_hit']].tolist()
    # Mean distance from each non-hit to the target group
    candidate_rows = []
    for ni in non_hit_idx:
        d_to_targets = [dist_full[ni, ti] for ti in target_idx_in_combined]
        candidate_rows.append({
            'EOS_id': combined.iloc[ni]['Metadata_Compound'],
            'Plate_origin': combined.iloc[ni]['Metadata_Plate'],
            'Activity_score': combined.iloc[ni]['Activity_final'],
            'mean_distance_to_targets': float(np.mean(d_to_targets)),
            'min_distance_to_targets': float(np.min(d_to_targets)),
        })
    cand = pd.DataFrame(candidate_rows).sort_values('mean_distance_to_targets').reset_index(drop=True)
    cand.head(20).to_csv(OUT_DIR / 'candidates_near_target_group.csv', index=False)
    log(f"  Top 20 candidates near {TARGET_LABEL} cluster saved.")
    log(f"  Top 5: {cand.head(5)['EOS_id'].tolist()}")


# ---------------------------------------------------------------------------
# 7. Section 4.7.3 - Phenotype validation (silhouette + Mann-Whitney)
# ---------------------------------------------------------------------------
section("STEP 7: Phenotype classification validation")

hits_phen = combined[combined['is_hit']].copy().reset_index(drop=True)
hits_idx_local = hits_phen.index.tolist()
hits_X_scaled = X_scaled[combined['is_hit'].values]
hits_dist = squareform(pdist(hits_X_scaled, metric='euclidean'))

phenotype_stats = []
phenotypes = ['shrunken', 'stable', 'expanded']

for phen in phenotypes:
    mask = (hits_phen['Phenotype'] == phen).values
    n = mask.sum()
    if n < 2:
        phenotype_stats.append({
            'phenotype': phen, 'n': int(n),
            'median_intra': np.nan, 'median_inter': np.nan,
            'p_value_mw': np.nan, 'lectura': 'sin potencia (n<2)'
        })
        continue
    intra_idx = np.where(mask)[0]
    inter_idx = np.where(~mask)[0]
    intra_dists = []
    for i, j in combinations(intra_idx, 2):
        intra_dists.append(hits_dist[i, j])
    inter_dists = []
    for i in intra_idx:
        for j in inter_idx:
            inter_dists.append(hits_dist[i, j])
    intra_dists = np.array(intra_dists)
    inter_dists = np.array(inter_dists)
    if len(intra_dists) > 0:
        u_stat, p_mw = mannwhitneyu(intra_dists, inter_dists, alternative='less')
    else:
        p_mw = np.nan
    if p_mw < 0.001:
        lectura = 'agrupa fuertemente'
    elif p_mw < 0.05:
        lectura = 'agrupa significativamente'
    else:
        lectura = 'no agrupa'
    phenotype_stats.append({
        'phenotype': phen, 'n': int(n),
        'median_intra': float(np.median(intra_dists)) if len(intra_dists) else np.nan,
        'median_inter': float(np.median(inter_dists)) if len(inter_dists) else np.nan,
        'p_value_mw': float(p_mw) if not np.isnan(p_mw) else None,
        'lectura': lectura,
    })

# Silhouette overall + permutation null
phen_labels = hits_phen['Phenotype'].values
unique_phen = [p for p in np.unique(phen_labels) if p != 'unknown']
if len(unique_phen) >= 2 and len(phen_labels) >= 4:
    label_map = {p: i for i, p in enumerate(unique_phen)}
    int_labels = np.array([label_map.get(p, -1) for p in phen_labels])
    valid = int_labels >= 0
    if valid.sum() >= 4 and len(np.unique(int_labels[valid])) >= 2:
        sil_obs = silhouette_score(hits_X_scaled[valid], int_labels[valid])
    else:
        sil_obs = np.nan
        valid = None

    n_perm_sil = min(1000, N_PERM // 10)
    sil_null = []
    if not np.isnan(sil_obs):
        labels_to_shuffle = int_labels[valid].copy()
        for _ in range(n_perm_sil):
            shuffled = RNG.permutation(labels_to_shuffle)
            sil_null.append(silhouette_score(hits_X_scaled[valid], shuffled))
        sil_null = np.array(sil_null)
        p_silhouette = ((sil_null >= sil_obs).sum() + 1) / (n_perm_sil + 1)
    else:
        sil_null = np.array([])
        p_silhouette = np.nan
else:
    sil_obs = np.nan
    sil_null = np.array([])
    p_silhouette = np.nan

phen_stats_df = pd.DataFrame(phenotype_stats)
phen_stats_df['silhouette_observed'] = sil_obs
phen_stats_df['silhouette_p_value'] = p_silhouette
phen_stats_df.to_csv(OUT_DIR / 'phenotype_validation_stats.csv', index=False)

log(f"  Silhouette observed: {sil_obs:.4f}")
if not np.isnan(p_silhouette):
    log(f"  Silhouette permutation p-value: {p_silhouette:.4f}")
for s in phenotype_stats:
    pmw = s['p_value_mw']
    log(f"  {s['phenotype']:25s} n={s['n']:>2}  "
        f"intra={s['median_intra']:.2f}  inter={s['median_inter']:.2f}  "
        f"p={pmw if pmw is None else f'{pmw:.4f}'}  -> {s['lectura']}")


# ---------------------------------------------------------------------------
# 8. PLOTS
# ---------------------------------------------------------------------------
section("STEP 8: Generate plots")

# ---------- Figure 1: PCA of hits only with target triangle (4.7.1)
log("  Plot 1: PCA of 25 hits + target triangle")
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
hits_pca = X_pca[hits_mask]
hits_meta = combined[hits_mask].reset_index(drop=True)

# Panel A: by plate
ax = axes[0]
for plate in PLATES:
    m = (hits_meta['Metadata_Plate'] == plate).values
    ax.scatter(hits_pca[m, 0], hits_pca[m, 1], s=120,
               c=PLATE_COLORS[plate], alpha=0.75,
               edgecolors='black', linewidth=0.5,
               label=f'{plate} (n={m.sum()})')
target_mask_hits = hits_meta['is_target'].values
if target_mask_hits.sum() > 0:
    ax.scatter(hits_pca[target_mask_hits, 0], hits_pca[target_mask_hits, 1],
               marker='*', s=400, c='gold', edgecolors='black', linewidth=1.5,
               label=f'{TARGET_LABEL}', zorder=5)
ax.set_xlabel(f'PC1 ({var_explained[0]:.1f}%)')
ax.set_ylabel(f'PC2 ({var_explained[1]:.1f}%)')
ax.set_title('PCA of hits — by plate', fontsize=11)
ax.legend(loc='best', fontsize=8)
ax.grid(True, alpha=0.3)

# Panel B: by phenotype
ax = axes[1]
for phen, color in PHENOTYPE_COLORS.items():
    m = (hits_meta['Phenotype'] == phen).values
    if m.sum() > 0:
        ax.scatter(hits_pca[m, 0], hits_pca[m, 1], s=120,
                   c=color, alpha=0.75,
                   edgecolors='black', linewidth=0.5,
                   label=f'{phen} (n={m.sum()})')
if target_mask_hits.sum() > 0:
    ax.scatter(hits_pca[target_mask_hits, 0], hits_pca[target_mask_hits, 1],
               marker='*', s=400, c='gold', edgecolors='black', linewidth=1.5,
               label=f'{TARGET_LABEL}', zorder=5)
ax.set_xlabel(f'PC1 ({var_explained[0]:.1f}%)')
ax.set_ylabel(f'PC2 ({var_explained[1]:.1f}%)')
ax.set_title('PCA of hits — by phenotype', fontsize=11)
ax.legend(loc='best', fontsize=8)
ax.grid(True, alpha=0.3)

# Panel C: target triangle
ax = axes[2]
ax.scatter(hits_pca[:, 0], hits_pca[:, 1], s=80, c='lightgray',
           alpha=0.6, edgecolors='gray', linewidth=0.3)
if target_mask_hits.sum() >= 2:
    target_pca = hits_pca[target_mask_hits]
    target_eos = hits_meta.loc[target_mask_hits, 'Metadata_Compound'].values
    target_drugs = hits_meta.loc[target_mask_hits, 'Drug_name'].values
    # Connect with lines
    for i, j in combinations(range(len(target_pca)), 2):
        ax.plot([target_pca[i, 0], target_pca[j, 0]],
                [target_pca[i, 1], target_pca[j, 1]],
                'orange', linewidth=2, alpha=0.7, zorder=3)
    ax.scatter(target_pca[:, 0], target_pca[:, 1],
               marker='*', s=500, c='gold', edgecolors='red', linewidth=1.5,
               zorder=5)
    for k, (eos, drug) in enumerate(zip(target_eos, target_drugs)):
        label = drug if isinstance(drug, str) else eos
        ax.annotate(label, (target_pca[k, 0], target_pca[k, 1]),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=9, fontweight='bold', color='darkred')
ax.set_xlabel(f'PC1 ({var_explained[0]:.1f}%)')
ax.set_ylabel(f'PC2 ({var_explained[1]:.1f}%)')
ax.set_title(f'{TARGET_LABEL} group — connected', fontsize=11)
ax.grid(True, alpha=0.3)

plt.suptitle(f'PCA of robust hits (n={n_hits}) on {len(common_features)} common features',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT_DIR / 'pca_hits_only.png', dpi=120, bbox_inches='tight')
plt.close()
log(f"    Saved: pca_hits_only.png")


# ---------- Figure 2: Permutation test plots (4.7.1)
if proximity_results and random_means is not None:
    log("  Plot 2: Proximity permutation tests")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A: pair distance distribution
    ax = axes[0]
    ax.hist(hit_pair_distances, bins=30, color='lightblue',
            edgecolor='steelblue', alpha=0.7, label=f'All hit pairs (n={len(hit_pair_distances)})')
    for d in target_pairs_dist:
        ax.axvline(d, color='orange', linewidth=2, alpha=0.8)
    ax.axvline(np.median(hit_pair_distances), color='gray',
               linestyle='--', linewidth=1.2, label='Median (all hit pairs)')
    ax.axvline(target_mean_dist, color='red', linewidth=2,
               label=f'{TARGET_LABEL} mean dist')
    ax.set_xlabel('Euclidean distance (scaled feature space)')
    ax.set_ylabel('Number of pairs')
    ax.set_title(f'A. Distances between hit pairs\n'
                 f'(orange lines = {TARGET_LABEL} pairs)', fontsize=11)
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel B: permutation null distribution
    ax = axes[1]
    ax.hist(random_means, bins=40, color='lightblue',
            edgecolor='steelblue', alpha=0.7,
            label=f'Random {triplet_size}-tuples of hits')
    ax.axvline(np.mean(random_means), color='gray',
               linestyle=':', linewidth=1.2, label=f'Expected (random)')
    ax.axvline(target_mean_dist, color='red', linewidth=2,
               label=f'Observed ({TARGET_LABEL})')
    p_text = f'p = {p_vs_hits:.4f}' if not np.isnan(p_vs_hits) else 'p = N/A'
    ax.set_xlabel(f'Mean distance of {triplet_size}-tuple')
    ax.set_ylabel('Number of permutations')
    ax.set_title(f'B. Permutation test (n={len(random_means)})\n{p_text}',
                 fontsize=11)
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'¿Está {TARGET_LABEL} más cerca que el azar? '
                 f'(within hits)', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUT_DIR / f'proximity_test_{TARGET_LABEL}.png', dpi=120, bbox_inches='tight')
    plt.close()
    log(f"    Saved: proximity_test_{TARGET_LABEL}.png")


# ---------- Figures 3-5: Full landscape PCA / t-SNE / UMAP (4.7.2)
def plot_landscape(coords_2d, label, filename, var_exp=None):
    log(f"  Plot: {label} full landscape")
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    is_hit_arr = combined['is_hit'].values
    is_target_arr = combined['is_target'].values
    plate_arr = combined['Metadata_Plate'].values
    phen_arr = combined['Phenotype'].values
    activity_arr = combined['Activity_final'].values

    # Panel A: hit highlight
    ax = axes[0, 0]
    ax.scatter(coords_2d[~is_hit_arr, 0], coords_2d[~is_hit_arr, 1],
               s=8, c='lightgray', alpha=0.4, edgecolors='none')
    non_target_hits = is_hit_arr & ~is_target_arr
    ax.scatter(coords_2d[non_target_hits, 0], coords_2d[non_target_hits, 1],
               s=100, c='#E67E22', alpha=0.85,
               edgecolors='black', linewidth=0.5,
               label=f'Hit robusto (n={non_target_hits.sum()})')
    if is_target_arr.sum() > 0:
        ax.scatter(coords_2d[is_target_arr, 0], coords_2d[is_target_arr, 1],
                   marker='*', s=500, c='red', edgecolors='black', linewidth=1.5,
                   label=f'Hit {TARGET_LABEL} (n={is_target_arr.sum()})', zorder=5)
        for ti in target_idx_in_combined:
            drug = combined.iloc[ti]['Drug_name']
            label_text = drug if isinstance(drug, str) else combined.iloc[ti]['Metadata_Compound']
            ax.annotate(label_text, (coords_2d[ti, 0], coords_2d[ti, 1]),
                        xytext=(7, 7), textcoords='offset points',
                        fontsize=8, fontweight='bold', color='darkred',
                        bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='red', alpha=0.7))
    ax.set_title(f'A. {label} — paisaje completo ({len(combined)} compuestos)', fontsize=11)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel B: same with hit labels (drug names)
    ax = axes[0, 1]
    ax.scatter(coords_2d[~is_hit_arr, 0], coords_2d[~is_hit_arr, 1],
               s=8, c='lightgray', alpha=0.4, edgecolors='none')
    ax.scatter(coords_2d[is_hit_arr, 0], coords_2d[is_hit_arr, 1],
               s=80, c='#E67E22', alpha=0.85, edgecolors='black', linewidth=0.5)
    for hi in hits_idx:
        drug = combined.iloc[hi]['Drug_name']
        label_text = drug if isinstance(drug, str) else combined.iloc[hi]['Metadata_Compound']
        ax.annotate(label_text[:14], (coords_2d[hi, 0], coords_2d[hi, 1]),
                    xytext=(4, 4), textcoords='offset points',
                    fontsize=6.5, alpha=0.85)
    if is_target_arr.sum() > 0:
        ax.scatter(coords_2d[is_target_arr, 0], coords_2d[is_target_arr, 1],
                   marker='*', s=400, c='red', edgecolors='black', linewidth=1.5, zorder=5)
    ax.set_title(f'B. {label} — etiquetas de hits', fontsize=11)
    ax.grid(True, alpha=0.3)

    # Panel C: by plate
    ax = axes[1, 0]
    for plate in PLATES:
        m = plate_arr == plate
        ax.scatter(coords_2d[m, 0], coords_2d[m, 1],
                   s=10, c=PLATE_COLORS[plate], alpha=0.45,
                   edgecolors='none', label=f'{plate} (n={m.sum()})')
    if is_target_arr.sum() > 0:
        ax.scatter(coords_2d[is_target_arr, 0], coords_2d[is_target_arr, 1],
                   marker='*', s=400, c='red', edgecolors='black', linewidth=1.5,
                   label=TARGET_LABEL, zorder=5)
    ax.set_title(f'C. {label} — by plate', fontsize=11)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel D: colored by activity score
    ax = axes[1, 1]
    sc = ax.scatter(coords_2d[~is_hit_arr, 0], coords_2d[~is_hit_arr, 1],
                    s=10, c=activity_arr[~is_hit_arr], cmap='viridis',
                    alpha=0.5, edgecolors='none', vmin=0,
                    vmax=np.percentile(activity_arr, 98))
    ax.scatter(coords_2d[is_hit_arr, 0], coords_2d[is_hit_arr, 1],
               s=80, c=activity_arr[is_hit_arr], cmap='viridis',
               edgecolors='red', linewidth=0.8,
               vmin=0, vmax=np.percentile(activity_arr, 98))
    if is_target_arr.sum() > 0:
        ax.scatter(coords_2d[is_target_arr, 0], coords_2d[is_target_arr, 1],
                   marker='*', s=400, c='red', edgecolors='black', linewidth=1.5, zorder=5)
    plt.colorbar(sc, ax=ax, label='Activity score')
    ax.set_title(f'D. {label} — colored by activity', fontsize=11)
    ax.grid(True, alpha=0.3)

    if var_exp is not None:
        for ax_ in axes.flatten():
            ax_.set_xlabel(f'{label}1 ({var_exp[0]:.1f}%)' if 'PCA' in label
                           else f'{label}1')
            ax_.set_ylabel(f'{label}2 ({var_exp[1]:.1f}%)' if 'PCA' in label
                           else f'{label}2')
    else:
        for ax_ in axes.flatten():
            ax_.set_xlabel(f'{label}1')
            ax_.set_ylabel(f'{label}2')

    plt.suptitle(f'Paisaje fenotípico ({label}): {len(combined)} compuestos · '
                 f'{len(common_features)} features comunes',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUT_DIR / filename, dpi=120, bbox_inches='tight')
    plt.close()
    log(f"    Saved: {filename}")


plot_landscape(X_pca, 'PCA', 'pca_full_landscape.png', var_exp=var_explained)
plot_landscape(X_tsne, 't-SNE', 'tsne_full_landscape.png')
if UMAP_AVAILABLE:
    plot_landscape(X_umap, 'UMAP', 'umap_full_landscape.png')


# ---------- Figure 6: Phenotype validation (4.7.3)
if not np.isnan(sil_obs):
    log("  Plot 6: Phenotype validation")
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    hits_pca_only = X_pca[hits_mask]
    hits_tsne_only = X_tsne[hits_mask]

    # Panel A: PCA of hits by phenotype
    ax = axes[0, 0]
    for phen, color in PHENOTYPE_COLORS.items():
        m = (hits_phen['Phenotype'] == phen).values
        if m.sum() > 0:
            ax.scatter(hits_pca_only[m, 0], hits_pca_only[m, 1], s=110,
                       c=color, alpha=0.8, edgecolors='black', linewidth=0.5,
                       label=f'{phen} (n={m.sum()})')
    target_mask_phen = hits_phen['is_target'].values
    if target_mask_phen.sum() > 0:
        ax.scatter(hits_pca_only[target_mask_phen, 0], hits_pca_only[target_mask_phen, 1],
                   marker='*', s=400, facecolor='none', edgecolors='red', linewidth=1.5,
                   label=TARGET_LABEL, zorder=5)
    ax.set_xlabel(f'PC1 ({var_explained[0]:.1f}%)')
    ax.set_ylabel(f'PC2 ({var_explained[1]:.1f}%)')
    ax.set_title('A. PCA — coloreado por fenotipo', fontsize=11)
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel B: t-SNE of hits by phenotype
    ax = axes[0, 1]
    for phen, color in PHENOTYPE_COLORS.items():
        m = (hits_phen['Phenotype'] == phen).values
        if m.sum() > 0:
            ax.scatter(hits_tsne_only[m, 0], hits_tsne_only[m, 1], s=110,
                       c=color, alpha=0.8, edgecolors='black', linewidth=0.5,
                       label=f'{phen} (n={m.sum()})')
    if target_mask_phen.sum() > 0:
        ax.scatter(hits_tsne_only[target_mask_phen, 0], hits_tsne_only[target_mask_phen, 1],
                   marker='*', s=400, facecolor='none', edgecolors='red', linewidth=1.5,
                   zorder=5)
    ax.set_xlabel('t-SNE 1'); ax.set_ylabel('t-SNE 2')
    ax.set_title('B. t-SNE — coloreado por fenotipo', fontsize=11)
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel C: intra vs inter distances boxplots
    ax = axes[1, 0]
    box_data = []
    box_labels = []
    p_texts = []
    for s in phenotype_stats:
        if s['n'] < 2 or s['median_intra'] is None or np.isnan(s['median_intra']):
            continue
        phen = s['phenotype']
        mask = (hits_phen['Phenotype'] == phen).values
        intra_idx = np.where(mask)[0]
        inter_idx = np.where(~mask)[0]
        intra = [hits_dist[i, j] for i, j in combinations(intra_idx, 2)]
        inter = [hits_dist[i, j] for i in intra_idx for j in inter_idx]
        if len(intra) > 0:
            box_data.append(intra)
            box_labels.append(f'{phen}\nintra (n={len(intra)})')
        if len(inter) > 0:
            box_data.append(inter)
            box_labels.append(f'{phen}\ninter (n={len(inter)})')
        pmw = s['p_value_mw']
        p_texts.append(f"{phen}: p={pmw if pmw is None else f'{pmw:.4f}'}")
    if box_data:
        bp = ax.boxplot(box_data, tick_labels=box_labels, patch_artist=True)
        colors_box = []
        for i, lbl in enumerate(box_labels):
            colors_box.append('#A8E6CF' if 'intra' in lbl else '#CCCCCC')
        for patch, color in zip(bp['boxes'], colors_box):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)
    ax.set_ylabel('Distancia euclídea')
    ax.set_title('C. Distancias intra-fenotipo (verde) vs inter (gris)', fontsize=11)
    ax.tick_params(axis='x', labelsize=7)
    ax.grid(True, alpha=0.3)

    # Panel D: silhouette permutation
    ax = axes[1, 1]
    if len(sil_null) > 0:
        ax.hist(sil_null, bins=30, color='lightblue', edgecolor='steelblue',
                alpha=0.7, label=f'Etiquetas aleatorias (n={len(sil_null)})')
        ax.axvline(np.median(sil_null), color='gray', linestyle=':',
                   linewidth=1.2, label=f'Mediana null = {np.median(sil_null):.4f}')
        ax.axvline(sil_obs, color='red', linewidth=2,
                   label=f'Observado = {sil_obs:.4f}')
        ax.set_xlabel('Silhouette score')
        ax.set_ylabel('Número de permutaciones')
        ax.set_title(f'D. Silhouette: clasificación vs azar\np = {p_silhouette:.4f}',
                     fontsize=11)
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle('¿Agrupan los hits por fenotipo asignado?',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'phenotype_validation.png', dpi=120, bbox_inches='tight')
    plt.close()
    log("    Saved: phenotype_validation.png")


# ---------- Figure 7: Hits-only distance heatmap (4.7.4 part 1)
log("  Plot 7: 25 hits distance heatmap (Ward)")
if n_hits >= 4:
    Z = linkage(pdist(hits_X_scaled), method='ward')
    order = np.argsort(fcluster(Z, t=n_hits, criterion='maxclust'))
    # Better: use the leaves order from linkage
    from scipy.cluster.hierarchy import leaves_list
    order = leaves_list(Z)

    hits_dist_ord = hits_dist[np.ix_(order, order)]
    labels_ord = hits_phen.iloc[order]['Drug_name'].fillna(
        hits_phen.iloc[order]['Metadata_Compound']).values
    is_target_ord = hits_phen.iloc[order]['is_target'].values

    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(hits_dist_ord, cmap='RdBu_r', aspect='auto')
    ax.set_xticks(range(len(labels_ord)))
    ax.set_yticks(range(len(labels_ord)))
    label_colors = ['darkred' if t else 'black' for t in is_target_ord]
    label_weights = ['bold' if t else 'normal' for t in is_target_ord]
    ax.set_xticklabels(labels_ord, rotation=70, ha='right', fontsize=8)
    ax.set_yticklabels(labels_ord, fontsize=8)
    for i, (color, weight) in enumerate(zip(label_colors, label_weights)):
        ax.get_xticklabels()[i].set_color(color)
        ax.get_xticklabels()[i].set_fontweight(weight)
        ax.get_yticklabels()[i].set_color(color)
        ax.get_yticklabels()[i].set_fontweight(weight)
    plt.colorbar(im, ax=ax, label='Distancia euclídea')
    ax.set_title(f'Heatmap de distancias entre los {n_hits} hits\n'
                 f'(ordenado por clustering jerárquico Ward · '
                 f'rojo oscuro = miembro del grupo {TARGET_LABEL})',
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'hits_distance_heatmap.png', dpi=120, bbox_inches='tight')
    plt.close()
    log("    Saved: hits_distance_heatmap.png")


# ---------- Figure 8: Top 100 compounds heatmap (4.7.4 part 2)
log(f"  Plot 8: Top {args.top_n_heatmap} compounds heatmap")
top_n = min(args.top_n_heatmap, len(combined))
top_idx = combined['Activity_final'].nlargest(top_n).index.values
top_X = X_scaled[top_idx]
top_meta = combined.iloc[top_idx].reset_index(drop=True)

# Reorder rows by hierarchical clustering
Z_top = linkage(pdist(top_X), method='ward')
from scipy.cluster.hierarchy import leaves_list
row_order = leaves_list(Z_top)

# Reorder columns (features) by clustering too
Z_feat = linkage(pdist(top_X.T), method='ward')
col_order = leaves_list(Z_feat)

top_X_ord = top_X[row_order][:, col_order]
top_meta_ord = top_meta.iloc[row_order].reset_index(drop=True)

fig = plt.figure(figsize=(18, 11))
gs = fig.add_gridspec(2, 2, height_ratios=[0.05, 1], width_ratios=[1, 1],
                      hspace=0.05, wspace=0.15)

# Top color strip showing hit/target
ax_strip_a = fig.add_subplot(gs[0, 0])
ax_strip_b = fig.add_subplot(gs[0, 1])
strip_colors = []
for i, row in top_meta_ord.iterrows():
    if row['is_target']:
        strip_colors.append('#E74C3C')
    elif row['is_hit']:
        strip_colors.append('#E67E22')
    else:
        strip_colors.append('#ECF0F1')
strip_array = np.array([strip_colors])
ax_strip_a.imshow(np.arange(len(strip_colors)).reshape(1, -1),
                  aspect='auto', cmap=plt.cm.colors.ListedColormap(strip_colors))
ax_strip_a.set_xticks([])
ax_strip_a.set_yticks([])
ax_strip_a.set_title(f'A. Compuestos × features (z-score)', fontsize=11)
ax_strip_b.imshow(np.arange(len(strip_colors)).reshape(1, -1),
                  aspect='auto', cmap=plt.cm.colors.ListedColormap(strip_colors))
ax_strip_b.set_xticks([])
ax_strip_b.set_yticks([])
ax_strip_b.set_title(f'B. Distancias compuesto × compuesto', fontsize=11)

# Heatmaps
ax_a = fig.add_subplot(gs[1, 0])
im_a = ax_a.imshow(top_X_ord.T, aspect='auto', cmap='RdBu_r', vmin=-3, vmax=3)
ax_a.set_yticks([])
ax_a.set_xticks([])
ax_a.set_xlabel(f'Compuestos (n={top_n}, top por activity score)')
ax_a.set_ylabel(f'{len(common_features)} features (clustered)')
plt.colorbar(im_a, ax=ax_a, label='z-score', fraction=0.04)

ax_b = fig.add_subplot(gs[1, 1])
top_dist = squareform(pdist(top_X_ord))
im_b = ax_b.imshow(top_dist, aspect='auto', cmap='RdBu_r')
ax_b.set_xticks([])
ax_b.set_yticks([])
plt.colorbar(im_b, ax=ax_b, label='Distancia euclídea', fraction=0.04)

# Legend
legend_elements = [
    Patch(facecolor='#ECF0F1', edgecolor='gray', label='No-hit'),
    Patch(facecolor='#E67E22', edgecolor='black', label='Hit robusto'),
    Patch(facecolor='#E74C3C', edgecolor='black', label=f'Hit {TARGET_LABEL}'),
]
fig.legend(handles=legend_elements, loc='upper center',
           bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=9)
plt.suptitle(f'Top {top_n} compuestos por activity score · '
             f'ordenados por similitud fenotípica',
             fontsize=12, fontweight='bold', y=1.02)
plt.savefig(OUT_DIR / 'top100_compounds_heatmap.png', dpi=120, bbox_inches='tight')
plt.close()
log("    Saved: top100_compounds_heatmap.png")


# ---------------------------------------------------------------------------
# 9. Narrative summary
# ---------------------------------------------------------------------------
section("STEP 9: Write narrative summary")

summary_path = OUT_DIR / 'multivariate_summary.txt'
with open(summary_path, 'w', encoding='utf-8') as f:
    f.write(f"CP3D — Análisis multivariante (PCA / t-SNE / UMAP)\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"Compuestos analizados:    {len(combined)}\n")
    f.write(f"Hits robustos:            {n_hits}\n")
    f.write(f"Features comunes:         {len(common_features)} (intersección 3 placas)\n")
    f.write(f"Target group:             {TARGET_LABEL} ({len(TARGET_GROUP)} compuestos)\n")
    if TARGET_GROUP:
        for t in TARGET_GROUP:
            row = combined[combined['Metadata_Compound'] == t]
            if len(row) > 0:
                drug = row.iloc[0]['Drug_name']
                f.write(f"                          {t} = {drug}\n")
    f.write("\n")

    f.write("PARTE 1 — Proximidad del target group dentro de los hits (sec. 4.7.1)\n")
    f.write("-" * 70 + "\n")
    if proximity_results:
        f.write(f"  Pares dentro del grupo: {len(proximity_results['pair_distances'])}\n")
        for d, p in zip(proximity_results['pair_distances'],
                        proximity_results['pair_percentiles_within_hits']):
            f.write(f"    distancia = {d:.3f}  (percentil {p:.1f}% dentro de pares de hits)\n")
        f.write(f"  Distancia media observada: {proximity_results['mean_dist_target']:.3f}\n")
        f.write(f"  Mediana de distancias entre pares de hits: "
                f"{proximity_results['median_dist_hits_global']:.3f}\n")
        if proximity_results['p_vs_random_hit_triplets'] is not None:
            f.write(f"  p-value test permutación (vs tripletes aleatorios de hits): "
                    f"{proximity_results['p_vs_random_hit_triplets']:.4f}\n")
        if proximity_results['p_vs_random_library_triplets'] is not None:
            f.write(f"  p-value test permutación (vs library completa): "
                    f"{proximity_results['p_vs_random_library_triplets']:.4f}\n")
    else:
        f.write("  (no calculado)\n")
    f.write("\n")

    f.write("PARTE 2 — Validación de la clasificación fenotípica (sec. 4.7.3)\n")
    f.write("-" * 70 + "\n")
    f.write(f"  Silhouette score observado: {sil_obs:.4f}\n")
    if not np.isnan(p_silhouette):
        f.write(f"  Silhouette p-value (vs etiquetas aleatorias): {p_silhouette:.4f}\n")
    f.write("\n")
    for s in phenotype_stats:
        pmw = s['p_value_mw']
        pmw_str = f"{pmw:.4f}" if isinstance(pmw, float) else 'N/A'
        f.write(f"  {s['phenotype']:25s} n={s['n']:>2}  "
                f"intra mediana={s['median_intra']:.2f}  "
                f"inter mediana={s['median_inter']:.2f}  "
                f"p={pmw_str}  -> {s['lectura']}\n")
    f.write("\n")

    f.write("OUTPUTS GENERADOS\n")
    f.write("-" * 70 + "\n")
    for p in sorted(OUT_DIR.glob('*')):
        if p.is_file() and p.name != summary_path.name:
            f.write(f"  {p.name}  ({p.stat().st_size // 1024} KB)\n")

log(f"  Summary saved: {summary_path}")

section("DONE", char='#')
log(f"All outputs in: {OUT_DIR}")
