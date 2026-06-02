"""
14a_enrich_dili_pubchem.py
=============================
Enrich the FDA DILIrank dataset with PubChem InChIKey for each compound.

DILIrank ships with compound names only (no SMILES or InChIKey). Cross-matching
the EU-OPENSCREEN bioactive library by name is unreliable due to synonyms,
brand names, salt forms, etc. (~12% coverage). This script queries PubChem
once for each DILIrank entry and saves the resulting InChIKey table to a CSV
cache that the main analysis (script 14) consumes for InChIKey-based matching.

Inputs
------
data/external/DILIrank.xlsx (or .csv)

Outputs
-------
data/external/DILIrank_pubchem_enriched.csv
    DILIrank rows + PubChem_CID + InChIKey + match_confidence.

Implementation notes
--------------------
- Uses pubchempy (thin wrapper around PubChem PUG REST API).
- Caches results so re-running is essentially instant.
- Respects PubChem rate limits (~5 requests/sec).
- For names with no PubChem match, retries with stripped salt form.

Usage
-----
    pip install pubchempy
    python 14a_enrich_dili_pubchem.py
    python 14a_enrich_dili_pubchem.py --force      # re-query everything
"""

import argparse
import sys
import time
from pathlib import Path
import pandas as pd

BASE = Path(r'C:\Users\Ianezi\Documents\CP3D')
EXT = BASE / 'analysis' / 'data' / 'external'
OUT_CSV = EXT / 'DILIrank_pubchem_enriched.csv'

SALT_FORMS = ['HYDROCHLORIDE', 'SODIUM', 'CALCIUM', 'POTASSIUM', 'MESYLATE',
              'SUCCINATE', 'CITRATE', 'PHOSPHATE', 'SULFATE', 'TARTRATE',
              'MALEATE', 'BESYLATE', 'TOSYLATE', 'FUMARATE', 'ACETATE',
              'BROMIDE', 'CHLORIDE', 'NITRATE', 'IODIDE']

parser = argparse.ArgumentParser()
parser.add_argument('--dili_file', default=None,
                    help='DILIrank file path. Default: auto-detect.')
parser.add_argument('--force', action='store_true',
                    help='Re-query PubChem even for entries already in cache.')
parser.add_argument('--sleep', type=float, default=0.22,
                    help='Seconds between PubChem queries (default 0.22 = ~5 req/sec).')
args = parser.parse_args()

print(f"\n{'='*70}\n=== DILIrank PubChem enrichment ===\n{'='*70}\n")

# ---------------------------------------------------------------------------
# 1. Load DILIrank
# ---------------------------------------------------------------------------
if args.dili_file:
    dili_path = Path(args.dili_file)
else:
    cands = list(EXT.glob('DILIrank*.xlsx')) + list(EXT.glob('DILIrank*.csv'))
    cands = [c for c in cands if 'enriched' not in c.name.lower()]
    if not cands:
        print(f"  ERROR: no DILIrank file in {EXT}")
        sys.exit(1)
    dili_path = cands[0]
print(f"  Loading: {dili_path.name}")

def smart_load(path):
    """Auto-detect header row."""
    for h in range(5):
        try:
            if path.suffix.lower() == '.xlsx':
                df = pd.read_excel(path, header=h)
            else:
                df = pd.read_csv(path, header=h)
            cols_lower = ' '.join(str(c).lower() for c in df.columns)
            if sum(1 for p in ['compound', 'dili', 'concern']
                   if p in cols_lower) >= 2:
                return df, h
        except Exception:
            continue
    return pd.read_excel(path) if path.suffix.lower() == '.xlsx' else pd.read_csv(path), 0

dili, h = smart_load(dili_path)
print(f"  Header at row: {h}")
print(f"  Rows: {len(dili)}")
print(f"  Columns: {list(dili.columns)}")

# Identify name column
def find_col(df, patterns):
    for c in df.columns:
        cl = str(c).lower().strip()
        for p in patterns:
            if p.lower() in cl:
                return c
    return None

name_col = find_col(dili, ['CompoundName', 'Compound Name', 'Generic_Name', 'Drug Name', 'Name'])
if not name_col:
    print(f"  ERROR: no compound name column detected.")
    sys.exit(1)
print(f"  Compound name column: {name_col}")

# ---------------------------------------------------------------------------
# 2. Load cache
# ---------------------------------------------------------------------------
if OUT_CSV.exists() and not args.force:
    cache = pd.read_csv(OUT_CSV)
    print(f"\n  Existing cache: {OUT_CSV.name} ({len(cache)} entries)")
else:
    cache = pd.DataFrame(columns=list(dili.columns) +
                                    ['PubChem_CID', 'InChIKey', 'match_confidence'])
    print(f"\n  No cache yet — starting fresh.")

# ---------------------------------------------------------------------------
# 3. PubChem enrichment
# ---------------------------------------------------------------------------
try:
    import pubchempy as pcp
except ImportError:
    print(f"\n  ERROR: pubchempy not installed.")
    print(f"  Run: pip install pubchempy")
    sys.exit(1)

def strip_salt(name):
    name = str(name).upper().strip()
    for s in SALT_FORMS:
        if name.endswith(' ' + s):
            name = name[:-(len(s)+1)].strip()
    return name

def query_pubchem(name):
    """Return (CID, InChIKey, confidence_label) or (None, None, label)."""
    name = str(name).strip()
    if not name or name.lower() in ('nan', 'none'):
        return None, None, 'empty_name'
    # Try direct query
    try:
        results = pcp.get_compounds(name, 'name', listkey_count=1)
        if results:
            r = results[0]
            return r.cid, r.inchikey, 'direct'
    except Exception:
        pass
    # Try stripped salt form
    stripped = strip_salt(name)
    if stripped != name.upper():
        try:
            results = pcp.get_compounds(stripped, 'name', listkey_count=1)
            if results:
                r = results[0]
                return r.cid, r.inchikey, 'stripped_salt'
        except Exception:
            pass
    return None, None, 'no_match'

# Determine which rows need querying
existing_names = set(cache[name_col].astype(str)) if name_col in cache.columns else set()

to_query = dili[~dili[name_col].astype(str).isin(existing_names)].copy() \
            if not args.force else dili.copy()
print(f"  To query (not in cache): {len(to_query)}/{len(dili)}")

if len(to_query) == 0:
    print(f"  Cache complete. Nothing to do.")
else:
    print(f"  Estimated time: {len(to_query) * args.sleep / 60:.1f} min "
          f"(rate ~{1/args.sleep:.0f} req/sec)")
    new_rows = []
    start_time = time.time()
    n_direct = n_salt = n_fail = 0
    for i, (_, r) in enumerate(to_query.iterrows()):
        name = r[name_col]
        cid, ik, conf = query_pubchem(name)
        if conf == 'direct':
            n_direct += 1
        elif conf == 'stripped_salt':
            n_salt += 1
        else:
            n_fail += 1
        new_rows.append({**r.to_dict(),
                         'PubChem_CID': cid,
                         'InChIKey': ik,
                         'match_confidence': conf})
        if (i + 1) % 50 == 0 or (i + 1) == len(to_query):
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (len(to_query) - i - 1) / rate if rate > 0 else 0
            print(f"    [{i+1:>4}/{len(to_query)}]  direct={n_direct}, "
                  f"salt={n_salt}, fail={n_fail}, "
                  f"rate={rate:.1f}/s, ETA={eta:.0f}s", flush=True)
        time.sleep(args.sleep)

    # Merge with cache
    new_df = pd.DataFrame(new_rows)
    if len(cache):
        combined = pd.concat([cache, new_df], ignore_index=True)
    else:
        combined = new_df
    # Deduplicate by name (keep newest)
    combined = combined.drop_duplicates(subset=[name_col], keep='last')
    combined.to_csv(OUT_CSV, index=False)
    print(f"\n  Saved: {OUT_CSV} ({len(combined)} entries total)")

# ---------------------------------------------------------------------------
# 4. Final summary
# ---------------------------------------------------------------------------
final = pd.read_csv(OUT_CSV)
n_total = len(final)
n_with_ik = final['InChIKey'].notna().sum()
print(f"\n{'-'*70}\nFinal summary\n{'-'*70}")
print(f"  Total DILIrank entries:        {n_total}")
print(f"  Enriched with InChIKey:        {n_with_ik} ({100*n_with_ik/n_total:.1f}%)")
print(f"  match_confidence distribution:")
print(final['match_confidence'].value_counts().to_string())
print(f"\n{'='*70}\nDone. Now you can run: python 14_dili_predictor.py\n{'='*70}\n")
