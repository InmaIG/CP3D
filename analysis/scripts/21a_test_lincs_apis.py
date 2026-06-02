"""
21a_test_lincs_apis.py
=======================
Diagnostic: probe several LINCS L1000 / iLINCS / SigCom endpoints from your
local network and report which respond. Run this BEFORE the full validation
script (21) to determine which data source is reachable from your machine.

Tests 4 sources of LINCS L1000 data:

1. iLINCS API (Cincinnati) - POST with JSON body
   - http://www.ilincs.org/api/SignatureMeta/findCompoundPerturbations
   - http://www.ilincs.org/api/ilincsR/downloadSignature

2. SigCom LINCS API (Mt. Sinai / MaayanLab)
   - https://maayanlab.cloud/sigcom-lincs/metadata-api/

3. L1000FWD API (MaayanLab, older)
   - https://maayanlab.cloud/L1000FWD/

4. LINCS Data Portal 3 (DCIC Miami / MaayanLab)
   - https://maayanlab.cloud/sigcom-lincs/

Reports each test as PASS / FAIL with status code and snippet of response.

Usage
-----
    python 21a_test_lincs_apis.py
"""

import requests
import json
import sys

TIMEOUT = 15

def banner(title):
    print(f"\n{'='*70}\n{title}\n{'='*70}")

def test(name, fn):
    try:
        result = fn()
        if result:
            print(f"  PASS  {name}")
            return True
        else:
            print(f"  FAIL  {name}  (empty or invalid response)")
            return False
    except requests.RequestException as e:
        print(f"  FAIL  {name}  (network error: {e.__class__.__name__})")
        return False
    except Exception as e:
        print(f"  FAIL  {name}  (error: {e})")
        return False

# ---------------------------------------------------------------------------
# 1. iLINCS - POST method
# ---------------------------------------------------------------------------
banner("1. iLINCS (Cincinnati) — POST with JSON body")

def ilincs_find_compound_post():
    url = 'http://www.ilincs.org/api/SignatureMeta/findCompoundPerturbations'
    r = requests.post(url, json={'keywords': 'geldanamycin'}, timeout=TIMEOUT)
    print(f"    HTTP {r.status_code}")
    if r.ok and r.text:
        try:
            data = r.json()
            n = len(data) if isinstance(data, list) else 0
            print(f"    Response: {n} signatures, first item keys: "
                  f"{list(data[0].keys())[:5] if n else 'empty'}")
            return True
        except Exception as e:
            print(f"    JSON parse failed: {e}")
            print(f"    First 200 chars of response: {r.text[:200]}")
    else:
        print(f"    Response: {r.text[:200]}")
    return False

def ilincs_find_compound_get():
    url = 'http://www.ilincs.org/api/SignatureMeta/findCompoundPerturbations'
    r = requests.get(url, params={'keywords': 'geldanamycin'}, timeout=TIMEOUT)
    print(f"    HTTP {r.status_code}")
    if r.ok and r.text:
        try:
            data = r.json()
            n = len(data) if isinstance(data, list) else 0
            print(f"    Response: {n} signatures")
            return True
        except Exception:
            pass
    return False

def ilincs_signature_libraries():
    url = 'http://www.ilincs.org/api/SignatureLibraries'
    r = requests.get(url, timeout=TIMEOUT)
    print(f"    HTTP {r.status_code}")
    if r.ok:
        try:
            data = r.json()
            n = len(data) if isinstance(data, list) else 0
            print(f"    Response: {n} libraries listed")
            if n:
                print(f"    First library: {data[0]}")
            return True
        except Exception:
            pass
    return False

test("ilincs POST findCompoundPerturbations(geldanamycin)", ilincs_find_compound_post)
test("ilincs GET  findCompoundPerturbations(geldanamycin)", ilincs_find_compound_get)
test("ilincs GET  SignatureLibraries", ilincs_signature_libraries)

# ---------------------------------------------------------------------------
# 2. SigCom LINCS API (MaayanLab) - currently maintained, free
# ---------------------------------------------------------------------------
banner("2. SigCom LINCS API (MaayanLab) — modern replacement")

def sigcom_test_root():
    url = 'https://maayanlab.cloud/sigcom-lincs/metadata-api/signatures/count'
    r = requests.get(url, timeout=TIMEOUT)
    print(f"    HTTP {r.status_code}")
    if r.ok:
        print(f"    Response: {r.text[:200]}")
        return True
    return False

def sigcom_find_signature():
    url = 'https://maayanlab.cloud/sigcom-lincs/metadata-api/entities/find'
    payload = {
        'filter': {'where': {'meta.pert_name': 'geldanamycin'}, 'limit': 5}
    }
    r = requests.post(url, json=payload, timeout=TIMEOUT)
    print(f"    HTTP {r.status_code}")
    if r.ok:
        try:
            data = r.json()
            n = len(data) if isinstance(data, list) else 0
            print(f"    Response: {n} signatures found")
            if n:
                print(f"    First signature keys: {list(data[0].keys())[:6]}")
            return True
        except Exception as e:
            print(f"    JSON parse error: {e}")
    return False

test("sigcom GET   signatures/count", sigcom_test_root)
test("sigcom POST  entities/find geldanamycin", sigcom_find_signature)

# ---------------------------------------------------------------------------
# 3. L1000FWD (older, possibly retired)
# ---------------------------------------------------------------------------
banner("3. L1000FWD (older MaayanLab service)")

def l1000fwd_search():
    url = 'https://maayanlab.cloud/L1000FWD/synonyms/geldanamycin'
    r = requests.get(url, timeout=TIMEOUT)
    print(f"    HTTP {r.status_code}")
    return r.ok

test("L1000FWD synonyms(geldanamycin)", l1000fwd_search)

# ---------------------------------------------------------------------------
# 4. Bulk download options (no API needed)
# ---------------------------------------------------------------------------
banner("4. Direct bulk download options")

def test_clue_homepage():
    r = requests.get('https://clue.io/', timeout=TIMEOUT)
    print(f"    HTTP {r.status_code}")
    return r.ok

def test_geo_l1000():
    # GSE92742 - LINCS L1000 Phase II
    r = requests.get('https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE92742',
                     timeout=TIMEOUT)
    print(f"    HTTP {r.status_code}")
    return r.ok

test("CLUE.io homepage reachable", test_clue_homepage)
test("GEO GSE92742 page reachable", test_geo_l1000)

# ---------------------------------------------------------------------------
print(f"\n{'='*70}\nDIAGNOSTIC COMPLETE")
print(f"{'='*70}\n")
print("Next step recommendation:")
print("  - If any iLINCS endpoint PASSED -> rewrite script 21 with that endpoint")
print("  - If only sigcom-lincs works -> rewrite script 21 against SigCom API")
print("  - If nothing works -> manual download from CLUE.io required")
print("")
print("If you have a corporate proxy / firewall, you may need:")
print("  set HTTP_PROXY=http://your.proxy:8080")
print("  set HTTPS_PROXY=http://your.proxy:8080")
print("  before running the script.")
