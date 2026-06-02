"""
process_platemap_C2388.py
=========================
Procesa el plate map especifico de C2388 que tiene 4 replicas tecnicas
dispensadas en bloques de 5 columnas dentro de la misma placa fisica:
  - Replica 1: dest cols 1-5
  - Replica 2: dest cols 6-10
  - Replica 3: dest cols 11-15
  - Replica 4: dest cols 16-20
  - Cols 21-24: vacias

Usa el protocolo ECHO para mapear source -> destination y combina con
el platemap original para asignar compuestos.

Inputs:
- C:\\Users\\Ianezi\\Documents\\CP3D\\Platemaps\\Protocolo ECHO CP3D placa C2388.xlsx
- C:\\Users\\Ianezi\\Documents\\CP3D\\Platemaps\\Platemap_C2388.xlsx

Output:
- data\\platemaps\\processed\\platemap_C2388_processed.csv
   con columnas: Plate, Well, Compound, Concentration_uM, Well_type, Replicate

Uso:
    python process_platemap_C2388.py
"""

import pandas as pd
import re
from pathlib import Path

# ---------------- Config ----------------
INPUT_DIR = Path(r'C:\Users\Ianezi\Documents\CP3D\Platemaps')
OUTPUT_DIR = Path(r'C:\Users\Ianezi\Documents\CP3D\analysis\data\platemaps\processed')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PROTOCOL_FILE = INPUT_DIR / 'Protocolo ECHO CP3D placa C2388.xlsx'
PLATEMAP_FILE = INPUT_DIR / 'Platemap_C2388.xlsx'

PLATE_NAME = 'C2388'
COMPOUND_CONCENTRATION_uM = 10.0
DMSO_CONCENTRATION_uM = 0.0

# Source wells DMSO: A4-P4 and A5-P5
DMSO_SOURCE_WELLS = set(
    [f'{row}{col}' for row in 'ABCDEFGHIJKLMNOP' for col in [4, 5]]
)

# Replicate ranges (destination column)
REPLICATE_BLOCKS = {
    'R1': (1, 5),
    'R2': (6, 10),
    'R3': (11, 15),
    'R4': (16, 20),
}


def normalize_well(well_str):
    """A1 -> A01, P22 -> P22"""
    m = re.match(r'^([A-Z]+)(\d+)$', str(well_str).strip())
    if not m:
        raise ValueError(f"Well no reconocido: {well_str}")
    row, col = m.group(1), int(m.group(2))
    return f"{row}{col:02d}"


def get_replicate(well_str):
    """Determine which replicate a destination well belongs to."""
    m = re.match(r'^([A-Z]+)(\d+)$', str(well_str).strip())
    if not m:
        return None
    col = int(m.group(2))
    for rep, (lo, hi) in REPLICATE_BLOCKS.items():
        if lo <= col <= hi:
            return rep
    return None  # cols 21-24 are not in any replicate block


def main():
    print(f"\n{'='*70}\n=== PROCESS PLATEMAP: {PLATE_NAME} ===\n{'='*70}")
    
    # ---------------- Load protocol ----------------
    print(f"\nProtocol: {PROTOCOL_FILE}")
    if not PROTOCOL_FILE.exists():
        print(f"ERROR: No se encuentra {PROTOCOL_FILE}")
        return
    protocol = pd.read_excel(PROTOCOL_FILE)
    print(f"  Total transfers: {len(protocol)}")
    print(f"  Columns: {list(protocol.columns)}")
    
    # ---------------- Load original platemap ----------------
    print(f"\nPlatemap: {PLATEMAP_FILE}")
    if not PLATEMAP_FILE.exists():
        print(f"ERROR: No se encuentra {PLATEMAP_FILE}")
        return
    platemap = pd.read_excel(PLATEMAP_FILE)
    platemap = platemap.rename(columns={'Molecule name': 'Compound'})
    print(f"  Total compounds: {len(platemap)}")
    
    # Normalize well naming in platemap (A1 -> A01)
    platemap['Source_Well_norm'] = platemap['Well'].apply(normalize_well)
    
    # Build dictionary source_well -> compound name
    source_to_compound = dict(zip(platemap['Source_Well_norm'], platemap['Compound']))
    print(f"  Sample compounds: {list(source_to_compound.items())[:5]}")
    
    # ---------------- Process protocol ----------------
    print(f"\n{'-'*70}\nProcess protocol\n{'-'*70}")
    
    rows = []
    skipped_no_compound = []
    
    for _, row in protocol.iterrows():
        src_raw = row['Source Well']
        dst_raw = row['Destination Well']
        
        src = normalize_well(src_raw)
        dst = normalize_well(dst_raw)
        replicate = get_replicate(dst)
        
        if replicate is None:
            print(f"  WARNING: dest well {dst} not in any replicate block (skipped)")
            continue
        
        # Determine if DMSO or compound
        # Convert source format for DMSO check (A4 not A04 in our DMSO set)
        src_short = re.sub(r'^([A-Z]+)0*(\d+)$', r'\1\2', src)
        is_dmso = src_short in DMSO_SOURCE_WELLS
        
        if is_dmso:
            compound = 'DMSO'
            conc = DMSO_CONCENTRATION_uM
            wtype = 'control'
        else:
            compound = source_to_compound.get(src)
            if compound is None:
                # Try without zero-padding for backwards compat
                compound = source_to_compound.get(src_short)
            if compound is None:
                skipped_no_compound.append((src, dst))
                continue
            conc = COMPOUND_CONCENTRATION_uM
            wtype = 'compound'
        
        rows.append({
            'Plate': PLATE_NAME,
            'Well': dst,
            'Compound': compound,
            'Concentration_uM': conc,
            'Well_type': wtype,
            'Replicate': replicate,
            'Source_Well': src,
        })
    
    if skipped_no_compound:
        print(f"\n  WARNING: {len(skipped_no_compound)} transfers skipped (no compound found)")
        for src, dst in skipped_no_compound[:10]:
            print(f"    src={src} -> dst={dst}")
    
    df = pd.DataFrame(rows)
    
    # ---------------- Sort and save ----------------
    df['_row'] = df['Well'].str[0]
    df['_col'] = df['Well'].str[1:].astype(int)
    df = df.sort_values(['Replicate', '_col', '_row']).reset_index(drop=True)
    df = df.drop(columns=['_row', '_col'])
    
    out = OUTPUT_DIR / f'platemap_{PLATE_NAME}_processed.csv'
    df.to_csv(out, index=False)
    
    # ---------------- Summary ----------------
    print(f"\n{'-'*70}\nSummary\n{'-'*70}")
    print(f"\nTotal wells dispensed: {len(df)}")
    print(f"\nBy replicate:")
    for rep in ['R1', 'R2', 'R3', 'R4']:
        sub = df[df['Replicate']==rep]
        n_cpd = (sub['Well_type']=='compound').sum()
        n_dmso = (sub['Well_type']=='control').sum()
        unique_cpd = sub[sub['Well_type']=='compound']['Compound'].nunique()
        print(f"  {rep}: {len(sub)} wells = {n_cpd} compound ({unique_cpd} unique) + {n_dmso} DMSO")
    
    print(f"\nUnique compounds in plate: "
          f"{df[df['Well_type']=='compound']['Compound'].nunique()}")
    
    print(f"\nSaved: {out}")
    print(f"\nFirst 10 rows:")
    print(df.head(10).to_string(index=False))


if __name__ == '__main__':
    main()