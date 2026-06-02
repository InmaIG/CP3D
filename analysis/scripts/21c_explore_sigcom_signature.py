"""
21c_explore_sigcom_signature.py
================================
Now that we know /signatures/find works, explore:
1) How many signatures of geldanamycin exist and in which cell lines
2) How to download the per-gene z-scores (the actual data)
3) The URL pattern of the data API

Usage
-----
    python 21c_explore_sigcom_signature.py
"""

import requests
import json

META_BASE = 'https://maayanlab.cloud/sigcom-lincs/metadata-api'
TIMEOUT = 20

# ---------------------------------------------------------------------------
# 1. Search and inspect all geldanamycin signatures
# ---------------------------------------------------------------------------
print(f"\n{'='*70}\n1. Geldanamycin signatures in SigCom\n{'='*70}\n")
r = requests.post(f'{META_BASE}/signatures/find',
                   json={'filter': {'where': {'meta.pert_name': 'geldanamycin'},
                                     'limit': 100}},
                   timeout=TIMEOUT)
print(f"HTTP {r.status_code}")
if not r.ok:
    print(r.text[:300])
    raise SystemExit(1)

sigs = r.json()
print(f"Total signatures for geldanamycin: {len(sigs)}\n")

# Inspect structure of first signature in detail
first = sigs[0]
print(f"First signature top-level keys: {list(first.keys())}")
print(f"\nmeta keys: {list(first.get('meta', {}).keys())}\n")
print(f"Full meta of first sig:")
for k, v in first.get('meta', {}).items():
    vs = str(v)[:80]
    print(f"  {k}: {vs}")
print(f"\nSignature ID: {first.get('id')}")

# Tabulate cell lines available
print(f"\n{'-'*60}\nCell lines available for geldanamycin:\n{'-'*60}")
cell_lines = {}
for s in sigs:
    m = s.get('meta', {})
    cl = m.get('cell') or m.get('cell_line') or m.get('cell_id') or 'UNKNOWN'
    cell_lines[cl] = cell_lines.get(cl, 0) + 1
for cl, n in sorted(cell_lines.items(), key=lambda x: -x[1]):
    print(f"  {cl}: {n} signatures")

# ---------------------------------------------------------------------------
# 2. Try to find the data API URL to download a signature
# ---------------------------------------------------------------------------
print(f"\n{'='*70}\n2. Probing data download endpoints\n{'='*70}\n")
sig_id = first['id']
print(f"Using signature id: {sig_id}\n")

# Candidate URL patterns for data download
DATA_CANDIDATES = [
    ('GET',  f'https://maayanlab.cloud/sigcom-lincs/data-api/api/v1/signature/{sig_id}'),
    ('GET',  f'https://maayanlab.cloud/sigcom-lincs/data-api/api/v1/sig/{sig_id}'),
    ('GET',  f'https://maayanlab.cloud/sigcom-lincs/data-api/api/v1/signature/data?signature_id={sig_id}'),
    ('GET',  f'https://maayanlab.cloud/sigcom-lincs/data-api/api/v2/signatures/{sig_id}'),
    ('GET',  f'https://maayanlab.cloud/sigcom-lincs/data-api/api/{sig_id}'),
    ('POST', 'https://maayanlab.cloud/sigcom-lincs/data-api/api/v1/signature',
       {'signature_id': sig_id}),
    ('POST', 'https://maayanlab.cloud/sigcom-lincs/data-api/api/v1/data',
       {'signature_id': sig_id}),
    ('POST', 'https://maayanlab.cloud/sigcom-lincs/data-api/api/v1/enrich/ranktwosided',
       {'up_genes': ['HSPA1A', 'HSPA6'], 'down_genes': ['MYC']}),
    # Try plain LDP3 data file paths
    ('GET',  f'https://lincs-dcic.s3.amazonaws.com/LDP3/sigcom-data/{sig_id}'),
]

for entry in DATA_CANDIDATES:
    method, url = entry[0], entry[1]
    payload = entry[2] if len(entry) > 2 else None
    try:
        if method == 'GET':
            r = requests.get(url, timeout=TIMEOUT)
        else:
            r = requests.post(url, json=payload, timeout=TIMEOUT)
        print(f"  {method} {url[:90]}")
        print(f"     HTTP {r.status_code}, body[:100]: {r.text[:100]}")
        if r.ok and len(r.text) > 100:
            print(f"     >>> LOOKS LIKE A WORKING ENDPOINT")
    except Exception as e:
        print(f"  {method} {url[:90]}")
        print(f"     FAILED: {e}")

# ---------------------------------------------------------------------------
# 3. Look at signatures meta to find any "datafile" or "url" field
# ---------------------------------------------------------------------------
print(f"\n{'='*70}\n3. Searching for hints of data location in metadata\n{'='*70}\n")
for key, val in first.get('meta', {}).items():
    s = str(val)
    if any(hint in s.lower() for hint in ['http', '.gctx', 's3.', 'sigcom-data',
                                            'signature_id']):
        print(f"  meta.{key}: {s[:200]}")

# Also check top-level fields
for key, val in first.items():
    if key == 'meta':
        continue
    s = str(val)
    if any(hint in s.lower() for hint in ['http', '.gctx', 's3.']):
        print(f"  {key}: {s[:200]}")

print(f"\n{'='*70}\nExploration complete.\n{'='*70}")
