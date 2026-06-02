"""
21d_inspect_tsv.py
====================
Inspect the raw structure of a LINCS L1000 TSV signature file from S3
to diagnose why the parsed values look like noise (~0.01).

Usage
-----
    python 21d_inspect_tsv.py
"""

import io
import requests
import pandas as pd

# Use geldanamycin HepG2 from our previous successful query
# Get the persistent_id by re-querying SigCom
META_URL = 'https://maayanlab.cloud/sigcom-lincs/metadata-api/signatures/find'

print("\n=== Step 1: get one HEPG2 geldanamycin signature ===")
r = requests.post(META_URL,
                   json={'filter': {'where': {'meta.pert_name': 'geldanamycin'},
                                     'limit': 200}},
                   timeout=30)
sigs = r.json()
# Find one with cell_line HEPG2, dose 10 uM, time 24 h
chosen = None
for s in sigs:
    m = s.get('meta', {})
    if (m.get('cell_line', '').upper() == 'HEPG2'
        and '10' in str(m.get('pert_dose', ''))
        and '24' in str(m.get('pert_time', ''))):
        chosen = s
        break
if chosen is None:
    chosen = sigs[0]
m = chosen['meta']
url = m['persistent_id']
print(f"  Selected: cell={m['cell_line']}, dose={m['pert_dose']}, time={m['pert_time']}")
print(f"  URL: {url}\n")

print("=== Step 2: download raw TSV ===")
r = requests.get(url, timeout=30)
print(f"  HTTP {r.status_code}, size: {len(r.text)} chars\n")

print("=== Step 3: first 20 lines of raw content ===")
print("-" * 60)
for i, line in enumerate(r.text.split('\n')[:20]):
    print(f"  Line {i}: {line[:140]}")
print("-" * 60)

print("\n=== Step 4: parse and inspect columns ===")
df = pd.read_csv(io.StringIO(r.text), sep='\t')
print(f"  Shape: {df.shape}")
print(f"  Columns: {list(df.columns)}")
print(f"\n  Data types:")
print(df.dtypes.to_string())
print(f"\n  First 5 rows:")
print(df.head().to_string())
print(f"\n  Numeric column statistics:")
for c in df.select_dtypes(include='number').columns:
    print(f"    {c}: min={df[c].min():.3f}, max={df[c].max():.3f}, "
          f"mean={df[c].mean():.3f}, abs_max={df[c].abs().max():.3f}")

print("\n=== Step 5: find HSPA1A specifically ===")
# Try various gene-name columns
gene_candidate_cols = [c for c in df.columns
                        if any(kw in str(c).lower()
                                for kw in ['gene', 'symbol', 'name', 'id'])]
print(f"  Possible gene columns: {gene_candidate_cols}")

# For each possible gene column, search HSPA1A
for col in gene_candidate_cols:
    vals = df[col].astype(str).str.upper()
    matches = df[vals == 'HSPA1A']
    if not matches.empty:
        print(f"\n  Found HSPA1A in column '{col}':")
        print(matches.to_string())

# Also search columns that look like ID strings
for col in df.columns:
    if col in gene_candidate_cols:
        continue
    sample = str(df[col].iloc[0])
    if 'HSPA' in sample.upper() or sample.upper() in ['HSPA1A', 'HSPA1B']:
        print(f"\n  Found HSPA gene name in column '{col}'")
