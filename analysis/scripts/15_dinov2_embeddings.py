"""
15_dinov2_embeddings.py
========================
Compute DINOv2 ViT-S/14 morphological embeddings for all 3D HepG2 spheroid
maximum-intensity-projection (MIP) images of the CP3D screen.

Each of the four Cell Painting channels (ER, AGP, MITO, DNA) is processed
independently through DINOv2 as a grayscale image replicated to RGB. The
resulting 384-dimensional embeddings are concatenated per well into a
4 x 384 = 1536-dimensional descriptor that captures channel-specific
morphological information in a feature space learned by self-supervised
pre-training on 142 million natural images (Oquab et al., arXiv 2304.07193).
This embedding is fully independent of the CellProfiler features used by the
main CP3D analysis pipeline and serves as an orthogonal validation of the
phenotypic clusters reported in this study (e.g. the HSP90 ansamycin cluster).

Reference
---------
Oquab M, Darcet T, Moutakanni T, et al. DINOv2: Learning Robust Visual
Features without Supervision. arXiv:2304.07193 (2023).

Channel mapping (this study)
----------------------------
ch1 = ER  (concanavalin A + WGA staining of endoplasmic reticulum)
ch2 = AGP (phalloidin staining of F-actin, Golgi, plasma membrane)
ch3 = MITO (MitoTracker staining of mitochondria)
ch4 = DNA (Hoechst 33342 staining of nuclei)

Input
-----
Y:/CELL_PAINTING_2024_EXPORT/CP3D/<YYYYMMDD>_AKURA_HEPG2_C####R#_72h_CP/Images MIP/
  r##c##f##p##-ch#sk1fk1fl1.tiff   (Harmony export convention)
data/platemaps/processed/platemap_all_combined.csv
data/annotated/cp3d_library_annotated.csv

Output
------
data/embeddings/embeddings_per_well_channel.parquet   (long format, 1 row per channel)
data/embeddings/embeddings_per_well.parquet           (wide, 1536-dim per well)
data/embeddings/embeddings_per_compound.parquet       (wide, 1536-dim per EOS_id)
data/embeddings/processing_log.txt

Usage
-----
    python 15_dinov2_embeddings.py
    python 15_dinov2_embeddings.py --batch_size 8         # if OOM
    python 15_dinov2_embeddings.py --filter_plates C2387  # subset
"""

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
import tifffile
from tqdm import tqdm
from transformers import AutoModel, AutoImageProcessor

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_PROJECT = Path(r'C:\Users\Ianezi\Documents\CP3D')
MIP_ROOT = Path(r'Y:\CELL_PAINTING_2024_EXPORT\CP3D')
ANALYSIS = BASE_PROJECT / 'analysis'
EMB_DIR = ANALYSIS / 'data' / 'embeddings'
EMB_DIR.mkdir(parents=True, exist_ok=True)
LIB_FILE = ANALYSIS / 'data' / 'annotated' / 'cp3d_library_annotated.csv'
PLATEMAP = ANALYSIS / 'data' / 'platemaps' / 'processed' / 'platemap_all_combined.csv'
LOG_FILE = EMB_DIR / 'processing_log.txt'

CHANNEL_MAP = {'ch1': 'ER', 'ch2': 'AGP', 'ch3': 'MITO', 'ch4': 'DNA'}

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--model', default='facebook/dinov2-small',
                    help='HuggingFace model identifier (default: dinov2-small, 22M params).')
parser.add_argument('--batch_size', type=int, default=16,
                    help='Inference batch size (default 16). Reduce to 8 if CUDA OOM.')
parser.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu'])
parser.add_argument('--filter_plates', nargs='+', default=None,
                    help='Optional: subset by plate IDs (e.g. C2386 C2387).')
parser.add_argument('--checkpoint_every', type=int, default=2000,
                    help='Save partial results every N images processed.')
args = parser.parse_args()

device = 'cuda' if args.device == 'auto' and torch.cuda.is_available() else args.device
if device == 'auto':
    device = 'cpu'

# ---------------------------------------------------------------------------
# Logging helper
# ---------------------------------------------------------------------------
log_lines = []
def log(msg):
    print(msg)
    log_lines.append(msg)

log(f"\n{'='*70}")
log(f"=== DINOv2 multi-channel embeddings — 3D HepG2 Cell Painting MIPs ===")
log(f"{'='*70}\n")
log(f"  Device:       {device}")
log(f"  Model:        {args.model}")
log(f"  Batch size:   {args.batch_size}")
log(f"  Channel mode: 4 channels processed independently (concat 4 x 384 = 1536 dim)")
log(f"  Output:       {EMB_DIR}")

# ---------------------------------------------------------------------------
# 1. Discover MIP folders
# ---------------------------------------------------------------------------
log(f"\n{'-'*70}\n1. Discovering MIP folders\n{'-'*70}")
if not MIP_ROOT.exists():
    log(f"  ERROR: MIP root not accessible: {MIP_ROOT}")
    sys.exit(1)

# Regex captures either single replicate (R1, R2, ...) or multi (R1-R4) for the
# special C2388 design (4 technical replicates within one physical plate).
folder_pat = re.compile(r'(\d{8})_AKURA_HEPG2_(C\d+)(R\d+(?:-R\d+)?)_72h_CP', re.I)
folders = []
for fold in MIP_ROOT.iterdir():
    if not fold.is_dir():
        continue
    m = folder_pat.match(fold.name)
    if not m:
        continue
    date_str, plate, replicate_token = m.groups()
    if args.filter_plates and plate not in args.filter_plates:
        continue
    mip_dir = fold / 'Images MIP'
    if mip_dir.exists():
        is_multi = '-' in replicate_token
        folders.append({'date': date_str, 'plate': plate,
                        'replicate_token': replicate_token,
                        'is_multi_replicate': is_multi,
                        'folder_name': fold.name, 'path': mip_dir})

folders = sorted(folders, key=lambda f: (f['plate'], f['replicate_token']))
log(f"  Found {len(folders)} MIP folders:")
for f in folders:
    flag = ' [multi-replicate by column blocks]' if f['is_multi_replicate'] else ''
    log(f"    {f['plate']}{f['replicate_token']} ({f['date']}): {f['folder_name']}{flag}")

if not folders:
    log(f"  ERROR: no folders match the pattern {folder_pat.pattern}")
    sys.exit(1)

# ---------------------------------------------------------------------------
# 2. Index TIFF files and parse Harmony naming
# ---------------------------------------------------------------------------
log(f"\n{'-'*70}\n2. Indexing TIFF files\n{'-'*70}")
file_pat = re.compile(r'r(\d+)c(\d+)f(\d+)p(\d+)-ch(\d+)', re.I)

def well_name(r, c):
    return f"{chr(ord('A') + r - 1)}{c:02d}"

def replicate_from_column(col):
    """For C2388 multi-replicate plate: assign R1-R4 based on destination column.
    Columns 21-24 are empty in C2388 design."""
    if 1 <= col <= 5:   return 'R1'
    if 6 <= col <= 10:  return 'R2'
    if 11 <= col <= 15: return 'R3'
    if 16 <= col <= 20: return 'R4'
    return None

records = []
n_skipped_empty_cols = 0
for f in folders:
    plate, replicate_token = f['plate'], f['replicate_token']
    mip_dir = f['path']
    is_multi = f['is_multi_replicate']
    try:
        tiff_files = list(mip_dir.glob('*.tiff')) + list(mip_dir.glob('*.tif'))
    except Exception as e:
        log(f"    {plate}{replicate_token}: error listing — {e}")
        continue
    n_indexed, n_skipped = 0, 0
    for fp in tiff_files:
        m = file_pat.match(fp.name)
        if not m:
            continue
        r, c, field, plane, ch = map(int, m.groups())
        # Resolve actual replicate
        if is_multi:
            actual_replicate = replicate_from_column(c)
            if actual_replicate is None:
                n_skipped += 1     # columns 21-24 in C2388 are empty
                continue
        else:
            actual_replicate = replicate_token
        records.append({
            'plate': plate, 'replicate': actual_replicate,
            'folder_token': replicate_token,   # bookkeeping
            'well': well_name(r, c), 'row': r, 'col': c,
            'field': field, 'plane': plane,
            'channel': f'ch{ch}',
            'channel_name': CHANNEL_MAP.get(f'ch{ch}', f'ch{ch}'),
            'path': str(fp),
        })
        n_indexed += 1
    n_skipped_empty_cols += n_skipped
    msg = f"  {plate}{replicate_token}: {n_indexed} TIFFs indexed"
    if n_skipped:
        msg += f" ({n_skipped} skipped, empty cols 21-24)"
    log(msg)
if n_skipped_empty_cols:
    log(f"  Total skipped (empty cols 21-24 in multi-replicate plate): {n_skipped_empty_cols}")

files_df = pd.DataFrame(records)
log(f"\n  Total TIFFs: {len(files_df)}")
log(f"  Unique (plate, replicate, well): "
    f"{files_df.groupby(['plate','replicate','well']).ngroups}")
log(f"  Channels:")
for ch, n in files_df['channel'].value_counts().items():
    log(f"    {ch} ({CHANNEL_MAP.get(ch, '?')}): {n}")

# Verify single field, single plane (sanity check)
fields_per = files_df.groupby(['plate','replicate','well','channel'])['field'].nunique()
planes_per = files_df.groupby(['plate','replicate','well','channel'])['plane'].nunique()
log(f"  Fields per (plate, replicate, well, channel): {fields_per.unique().tolist()}")
log(f"  Planes per (plate, replicate, well, channel): {planes_per.unique().tolist()}")

# ---------------------------------------------------------------------------
# 3. Cross-reference with platemap
# ---------------------------------------------------------------------------
log(f"\n{'-'*70}\n3. Cross-referencing with platemap\n{'-'*70}")
if not PLATEMAP.exists():
    log(f"  ERROR: platemap not found: {PLATEMAP}")
    sys.exit(1)
pm = pd.read_csv(PLATEMAP)
log(f"  Platemap rows: {len(pm)}")
log(f"  Columns: {list(pm.columns)}")
# Normalize column names
pm = pm.rename(columns={'Plate': 'plate', 'Well': 'well',
                          'Compound': 'EOS_id', 'Well_type': 'well_type'})
pm = pm[['plate', 'well', 'EOS_id', 'well_type']].drop_duplicates(['plate', 'well'])

# Merge
files_df = files_df.merge(pm, on=['plate', 'well'], how='left')
n_mapped = files_df['EOS_id'].notna().sum()
log(f"  Files mapped to compound/control: {n_mapped}/{len(files_df)} "
    f"({100*n_mapped/len(files_df):.1f}%)")
log(f"  Well-type distribution:")
for wt, n in files_df['well_type'].fillna('UNMAPPED').value_counts().items():
    log(f"    {wt}: {n}")

# ---------------------------------------------------------------------------
# 4. Load DINOv2 model
# ---------------------------------------------------------------------------
log(f"\n{'-'*70}\n4. Loading DINOv2 model\n{'-'*70}")
model = AutoModel.from_pretrained(args.model).to(device)
model.eval()
processor = AutoImageProcessor.from_pretrained(args.model)
EMB_DIM = model.config.hidden_size
log(f"  Model:       {args.model}")
log(f"  Hidden dim:  {EMB_DIM}")
log(f"  Params:      {sum(p.numel() for p in model.parameters())/1e6:.1f} M")
log(f"  Device:      {next(model.parameters()).device}")

# ---------------------------------------------------------------------------
# 5. TIFF loading helper
# ---------------------------------------------------------------------------
def load_tiff_as_rgb(path):
    """Load a 16-bit grayscale TIFF, percentile-normalize, return PIL RGB."""
    img = tifffile.imread(path)
    if img.ndim == 3:
        img = img[..., 0] if img.shape[-1] <= 4 else img[0]
    # Robust percentile normalization (1-99%)
    lo, hi = np.percentile(img, [1, 99])
    img_norm = np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1)
    img_uint8 = (img_norm * 255).astype(np.uint8)
    rgb = np.stack([img_uint8] * 3, axis=-1)
    return Image.fromarray(rgb)

# ---------------------------------------------------------------------------
# 6. Inference (per-channel, batched)
# ---------------------------------------------------------------------------
log(f"\n{'-'*70}\n5. Computing DINOv2 embeddings\n{'-'*70}")

ckpt = EMB_DIR / 'checkpoint_per_well_channel.parquet'
if ckpt.exists():
    existing = pd.read_parquet(ckpt)
    done_keys = set(zip(existing['plate'], existing['replicate'],
                          existing['well'], existing['channel']))
    log(f"  Found checkpoint with {len(existing)} entries done; resuming.")
else:
    existing = pd.DataFrame()
    done_keys = set()

# We want one embedding per (plate, replicate, well, channel)
unique_keys = files_df[['plate', 'replicate', 'well', 'channel']].drop_duplicates()
todo_mask = ~unique_keys.set_index(['plate','replicate','well','channel']).index.isin(done_keys)
todo = unique_keys[todo_mask].reset_index(drop=True)
log(f"  Total keys: {len(unique_keys)}; remaining to process: {len(todo)}")

# Build a fast lookup: (plate, replicate, well, channel) -> path (take field=1, plane=any)
path_lookup = (files_df.sort_values(['field', 'plane'])
                       .drop_duplicates(['plate', 'replicate', 'well', 'channel'])
                       .set_index(['plate', 'replicate', 'well', 'channel'])['path']
                       .to_dict())

new_rows = list(existing.to_dict('records'))
t_start = time.time()
n_failed = 0

with torch.no_grad():
    n_batches = (len(todo) + args.batch_size - 1) // args.batch_size
    pbar = tqdm(range(0, len(todo), args.batch_size), total=n_batches,
                 desc='DINOv2', unit='batch')
    for batch_start in pbar:
        batch_rows = todo.iloc[batch_start:batch_start + args.batch_size]
        keys, imgs = [], []
        for _, row in batch_rows.iterrows():
            k = (row['plate'], row['replicate'], row['well'], row['channel'])
            path = path_lookup.get(k)
            if path is None:
                continue
            try:
                pil = load_tiff_as_rgb(path)
                keys.append(k)
                imgs.append(pil)
            except Exception as e:
                n_failed += 1
                pbar.write(f"  load failed: {path} -- {e}")
        if not imgs:
            continue
        # Preprocess + forward
        try:
            inputs = processor(images=imgs, return_tensors='pt').to(device)
            outputs = model(**inputs)
            cls_embs = outputs.last_hidden_state[:, 0].cpu().numpy()
        except torch.cuda.OutOfMemoryError:
            pbar.write(f"  CUDA OOM at batch {batch_start}. Retry with smaller --batch_size.")
            sys.exit(1)
        # Collect
        for key, emb in zip(keys, cls_embs):
            plate, replicate, well, channel = key
            r = {'plate': plate, 'replicate': replicate, 'well': well,
                 'channel': channel,
                 'channel_name': CHANNEL_MAP.get(channel, channel)}
            for i, v in enumerate(emb):
                r[f'emb_{i}'] = float(v)
            new_rows.append(r)
        # Periodic checkpoint
        if (batch_start // args.batch_size + 1) % max(1, args.checkpoint_every // args.batch_size) == 0:
            pd.DataFrame(new_rows).to_parquet(ckpt, index=False)
            pbar.set_postfix(saved=len(new_rows), failed=n_failed)

# Final save (long format)
df_long = pd.DataFrame(new_rows)
out_long = EMB_DIR / 'embeddings_per_well_channel.parquet'
df_long.to_parquet(out_long, index=False)
elapsed = time.time() - t_start
log(f"\n  Inference complete.")
log(f"  Total embeddings produced: {len(df_long)}")
log(f"  Failed loads:              {n_failed}")
log(f"  Time elapsed:              {elapsed/60:.1f} min")
log(f"  Saved:                     {out_long}")

# ---------------------------------------------------------------------------
# 7. Reshape to wide (1 row per well, 1536 features)
# ---------------------------------------------------------------------------
log(f"\n{'-'*70}\n6. Reshaping to wide per-well format (4 channels concatenated)\n{'-'*70}")
emb_cols = [f'emb_{i}' for i in range(EMB_DIM)]
wide_parts = []
for ch, ch_name in CHANNEL_MAP.items():
    sub = df_long[df_long['channel'] == ch].copy()
    rename_map = {c: f'{ch_name}_{c}' for c in emb_cols}
    sub = sub[['plate', 'replicate', 'well'] + emb_cols].rename(columns=rename_map)
    wide_parts.append(sub.set_index(['plate', 'replicate', 'well']))
emb_wide = pd.concat(wide_parts, axis=1).reset_index()
emb_wide = emb_wide.merge(pm, on=['plate', 'well'], how='left')
out_wide = EMB_DIR / 'embeddings_per_well.parquet'
emb_wide.to_parquet(out_wide, index=False)
log(f"  Per-well wide table: {emb_wide.shape}")
log(f"  Saved: {out_wide}")

# ---------------------------------------------------------------------------
# 8. Aggregate to per-compound consensus
# ---------------------------------------------------------------------------
log(f"\n{'-'*70}\n7. Aggregating to per-compound consensus (median across wells)\n{'-'*70}")
feature_cols = [c for c in emb_wide.columns
                if any(c.startswith(f'{ch}_emb_') for ch in CHANNEL_MAP.values())]
compound_df = (emb_wide.dropna(subset=['EOS_id'])
                        .groupby('EOS_id')[feature_cols]
                        .median()
                        .reset_index())
out_compound = EMB_DIR / 'embeddings_per_compound.parquet'
compound_df.to_parquet(out_compound, index=False)
log(f"  Per-compound table: {compound_df.shape}")
log(f"  Feature columns: {len(feature_cols)}  (= {len(CHANNEL_MAP)} channels x {EMB_DIM} dims)")
log(f"  Unique compounds: {compound_df['EOS_id'].nunique()}")
log(f"  Saved: {out_compound}")

# ---------------------------------------------------------------------------
# Write log file
# ---------------------------------------------------------------------------
with open(LOG_FILE, 'w', encoding='utf-8') as f:
    f.write('\n'.join(log_lines))
log(f"\n  Log written to: {LOG_FILE}")
log(f"\n{'='*70}\nDone. Run 16_dinov2_analysis.py next to validate clusters.\n{'='*70}\n")
