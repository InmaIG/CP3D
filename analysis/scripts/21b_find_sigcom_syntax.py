"""
21b_find_sigcom_syntax.py
==========================
Probe SigCom LINCS API to find the correct query syntax for searching
compounds. Tries 8-10 variants and reports which works.

Usage
-----
    python 21b_find_sigcom_syntax.py
"""

import requests
import json

BASE = 'https://maayanlab.cloud/sigcom-lincs/metadata-api'
TIMEOUT = 20

def safe_post(url, payload):
    try:
        r = requests.post(url, json=payload, timeout=TIMEOUT)
        return r
    except Exception as e:
        return None

def safe_get(url, params=None):
    try:
        r = requests.get(url, params=params, timeout=TIMEOUT)
        return r
    except Exception as e:
        return None

print(f"\n{'='*70}\nSigCom LINCS — finding the right syntax\n{'='*70}\n")

# ---------------------------------------------------------------------------
# 1. First explore what fields exist in a sample signature
# ---------------------------------------------------------------------------
print(f"{'-'*70}\n1. Inspecting field structure of a random signature\n{'-'*70}")
r = safe_post(f'{BASE}/entities/find', {'limit': 1})
if r and r.ok:
    data = r.json()
    if isinstance(data, list) and len(data):
        sample = data[0]
        print(f"  Sample signature structure:")
        print(f"  Top-level keys: {list(sample.keys())}")
        if 'meta' in sample:
            print(f"  meta keys: {list(sample['meta'].keys())}")
            print(f"  Sample meta (truncated):")
            for k, v in list(sample['meta'].items())[:15]:
                vs = str(v)[:60]
                print(f"    {k}: {vs}")
    else:
        print(f"  Empty response: {r.text[:300]}")
else:
    print(f"  Request failed: {r.status_code if r else 'no response'}")

# ---------------------------------------------------------------------------
# 2. Try multiple ways to search for geldanamycin
# ---------------------------------------------------------------------------
print(f"\n{'-'*70}\n2. Trying multiple query syntaxes for 'geldanamycin'\n{'-'*70}")

queries = [
    ('1', 'meta.pert_name == geldanamycin (lowercase exact)',
     {'filter': {'where': {'meta.pert_name': 'geldanamycin'}, 'limit': 3}}),

    ('2', 'meta.pert_name == GELDANAMYCIN (uppercase)',
     {'filter': {'where': {'meta.pert_name': 'GELDANAMYCIN'}, 'limit': 3}}),

    ('3', 'meta.pert_iname == geldanamycin',
     {'filter': {'where': {'meta.pert_iname': 'geldanamycin'}, 'limit': 3}}),

    ('4', 'meta.pert_name regexp /geldanamycin/i',
     {'filter': {'where': {'meta.pert_name': {'regexp': '/geldanamycin/i'}},
                  'limit': 3}}),

    ('5', 'meta.pert_name like geldanamycin (loopback "like")',
     {'filter': {'where': {'meta.pert_name': {'like': 'geldanamycin'}},
                  'limit': 3}}),

    ('6', 'meta.compound_name == geldanamycin',
     {'filter': {'where': {'meta.compound_name': 'geldanamycin'}, 'limit': 3}}),

    ('7', 'meta.pert_id == BRD-K81473043 (geldanamycin Broad ID)',
     {'filter': {'where': {'meta.pert_id': 'BRD-K81473043'}, 'limit': 3}}),

    ('8', 'fulltext search via /search endpoint',
     None),  # Will use GET instead below

    ('9', 'top-level name == geldanamycin (no meta prefix)',
     {'filter': {'where': {'name': 'geldanamycin'}, 'limit': 3}}),

    ('10', 'Sigcom search-style: include text',
     {'filter': {'where': {'meta.pert_name': {'inq': ['geldanamycin',
                                                        'Geldanamycin',
                                                        'GELDANAMYCIN']}},
                  'limit': 3}}),
]

for i, name, payload in queries:
    print(f"\n  Test {i}: {name}")
    if payload is None:
        # GET variant for test 8
        r = safe_get(f'{BASE}/search', params={'q': 'geldanamycin'})
        if r:
            print(f"    HTTP {r.status_code}, body: {r.text[:200]}")
        else:
            print(f"    GET failed")
        continue
    r = safe_post(f'{BASE}/entities/find', payload)
    if r is None:
        print(f"    REQUEST FAILED")
        continue
    print(f"    HTTP {r.status_code}")
    if r.ok:
        try:
            data = r.json()
            n = len(data) if isinstance(data, list) else 0
            if n > 0:
                print(f"    ✓ FOUND {n} signatures")
                # Show a useful subset of the first one
                first = data[0]
                if 'meta' in first:
                    m = first['meta']
                    print(f"    Example fields of first match:")
                    interesting = ['pert_name', 'pert_iname', 'pert_id', 'cell_line',
                                    'cell_id', 'pert_idose', 'pert_itime', 'datatype']
                    for k in interesting:
                        if k in m:
                            print(f"      {k}: {m[k]}")
            else:
                print(f"    -- 0 matches")
        except Exception as e:
            print(f"    JSON error: {e}")
            print(f"    Body: {r.text[:200]}")
    else:
        print(f"    Body: {r.text[:200]}")

# ---------------------------------------------------------------------------
# 3. Try the /signatures endpoint instead of /entities
# ---------------------------------------------------------------------------
print(f"\n{'-'*70}\n3. Trying /signatures/find endpoint\n{'-'*70}")
for kw in ['geldanamycin', 'Geldanamycin', 'GELDANAMYCIN']:
    payload = {'filter': {'where': {'meta.pert_name': kw}, 'limit': 3}}
    r = safe_post(f'{BASE}/signatures/find', payload)
    if r:
        print(f"  /signatures/find pert_name={kw}: HTTP {r.status_code}, "
              f"body[:100]: {r.text[:100]}")

# ---------------------------------------------------------------------------
# 4. Check SigCom for available libraries / cell lines
# ---------------------------------------------------------------------------
print(f"\n{'-'*70}\n4. Listing available libraries\n{'-'*70}")
r = safe_post(f'{BASE}/libraries/find', {'filter': {'limit': 30}})
if r and r.ok:
    try:
        libs = r.json()
        if isinstance(libs, list):
            print(f"  Found {len(libs)} libraries (showing first 10):")
            for lib in libs[:10]:
                lib_name = lib.get('meta', {}).get('Name', lib.get('id', 'unknown'))
                count = lib.get('count', '?')
                print(f"    {lib_name}: {count} signatures")
    except Exception as e:
        print(f"  Parse error: {e}")
        print(f"  Body: {r.text[:300]}")

print(f"\n{'='*70}\nDone.\n{'='*70}\n")
print("Once we know the working syntax, the full script 21 can be rewritten.")
