"""
10_build_annotated_library.py
==============================
Construye una tabla unica con los 735 compuestos del screen 3D anotados con
quimica (SMILES, InChIKey) y MoA, listos para el cruce con JUMP-Cell Painting.

Entradas requeridas:
  - data/processed/consensus_C2386.csv, consensus_C2387.csv, consensus_C2388.csv
  - EOS_compounds_smiles.csv  (con al menos: EOS_id, SMILES)
  - EOS_compounds_MoA.csv     (con: EOS_id, Drug_name, Target, MoA, etc.)
  - results/hits_summary/all_hits_combined.csv  (para flag is_hit_3D)

Salida:
  - data/annotated/cp3d_library_annotated.csv
  - data/annotated/cp3d_library_annotated_audit.txt (que se cruzo, que no)

Uso:
    python 10_build_annotated_library.py
    python 10_build_annotated_library.py --recompute_inchikey   # forzar RDKit
"""

import argparse
import sys
from pathlib import Path
import pandas as pd

BASE = Path(r'C:\Users\Ianezi\Documents\CP3D')
ANALYSIS = BASE / 'analysis'
PROC = ANALYSIS / 'data' / 'processed'
OUT_DIR = ANALYSIS / 'data' / 'annotated'
OUT_DIR.mkdir(parents=True, exist_ok=True)

SMILES_FILE = BASE / 'EOS_compounds_smiles.csv'
MOA_FILE = BASE / 'EOS_compounds_MoA.csv'
HITS_FILE = ANALYSIS / 'results' / 'hits_summary' / 'all_hits_combined.csv'

CONSENSUS_FILES = {
    'C2386': PROC / 'consensus_C2386.csv',
    'C2387': PROC / 'consensus_C2387.csv',
    'C2388': PROC / 'consensus_C2388.csv',
}

parser = argparse.ArgumentParser()
parser.add_argument('--recompute_inchikey', action='store_true',
                    help='Forzar recalculo del InChIKey con RDKit desde SMILES')
args = parser.parse_args()

print(f"\n{'='*70}\n=== BUILD ANNOTATED LIBRARY (735 compuestos) ===\n{'='*70}")

# ---------------------------------------------------------------------
# 1. Cargar consensus 3D de las 3 placas
# ---------------------------------------------------------------------
print(f"\n{'-'*70}\n1. Cargar consensus 3D\n{'-'*70}")
compounds_3d = []
for plate, path in CONSENSUS_FILES.items():
    if not path.exists():
        print(f"  ERROR: no se encuentra {path}")
        sys.exit(1)
    df = pd.read_csv(path)
    # Quedarnos solo con compuestos (no DMSO)
    df = df[df['Metadata_Well_type'] == 'compound'].copy()
    df['Source_plate'] = plate
    print(f"  {plate}: {len(df)} compuestos")
    compounds_3d.append(df)

consensus = pd.concat(compounds_3d, ignore_index=True)
print(f"\n  Total filas consensus: {len(consensus)}")
print(f"  EOS_id unicos: {consensus['Metadata_Compound'].nunique()}")
consensus = consensus.rename(columns={'Metadata_Compound': 'EOS_id'})

# ---------------------------------------------------------------------
# 2. Cargar SMILES library
# ---------------------------------------------------------------------
print(f"\n{'-'*70}\n2. Cargar SMILES library\n{'-'*70}")
if not SMILES_FILE.exists():
    print(f"\n  ERROR: falta {SMILES_FILE}")
    print("  Asegurate de copiarlo a la raiz de CP3D antes de ejecutar.")
    sys.exit(1)

try:
    smiles_df = pd.read_csv(SMILES_FILE, encoding='latin-1')
except UnicodeDecodeError:
    smiles_df = pd.read_csv(SMILES_FILE, encoding='utf-8', errors='replace')

# Detectar columnas de forma robusta (case-insensitive)
def find_col(df, candidates):
    """Devuelve el nombre real de la primera columna que coincida con
    cualquier candidato (comparacion case-insensitive)."""
    lower_to_real = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_to_real:
            return lower_to_real[cand.lower()]
    return None

eos_col = find_col(smiles_df, ['EOS_id', 'EOS', 'eos', 'eos_id', 'eosid', 'compound_id'])
if eos_col is None:
    print(f"  ERROR: columna EOS no encontrada. Columnas: {list(smiles_df.columns)}")
    sys.exit(1)

smiles_col = find_col(smiles_df, ['SMILES', 'canonical_smiles'])
if smiles_col is None:
    print(f"  ERROR: columna SMILES no encontrada. Columnas: {list(smiles_df.columns)}")
    sys.exit(1)

inchikey_col = find_col(smiles_df, ['InChIKey', 'InChI_Key', 'inchi_key'])

smiles_df = smiles_df.rename(columns={eos_col: 'EOS_id', smiles_col: 'SMILES'})
if inchikey_col:
    smiles_df = smiles_df.rename(columns={inchikey_col: 'InChIKey'})

n_rows_raw = len(smiles_df)
# Deduplicar: nos quedamos con un registro por EOS_id (el library puede traer
# multiples rows si incluye info de platemap)
smiles_df = smiles_df.drop_duplicates(subset=['EOS_id'], keep='first').reset_index(drop=True)
print(f"  SMILES library: {n_rows_raw} rows brutas -> {len(smiles_df)} compuestos unicos por EOS_id")
print(f"  Columnas detectadas: EOS_id={eos_col}, SMILES={smiles_col}, InChIKey={inchikey_col or '(falta, calculo con RDKit)'}")

# ---------------------------------------------------------------------
# 3. Calcular InChIKey canonical con RDKit (siempre — para tener una version
# normalizada que pueda matchear con JUMP-CP independientemente del estado
# de protonacion/estereo del archivo original)
# ---------------------------------------------------------------------
print(f"\n{'-'*70}\n3. Calcular InChIKey canonical con RDKit\n{'-'*70}")
try:
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
except ImportError:
    print("  AVISO: rdkit no instalado. InChIKey_rdkit no se computa.")
    print("         pip install rdkit  para habilitarlo")
    Chem = None

if Chem is not None:
    def smiles_to_inchikey(smi):
        try:
            mol = Chem.MolFromSmiles(str(smi))
            if mol is None:
                return None
            return Chem.MolToInchiKey(mol)
        except Exception:
            return None

    smiles_df['InChIKey_rdkit'] = smiles_df['SMILES'].apply(smiles_to_inchikey)
    n_ok = smiles_df['InChIKey_rdkit'].notna().sum()
    print(f"  InChIKey_rdkit (canonical desde SMILES): {n_ok}/{len(smiles_df)}")

if inchikey_col is None:
    # Si el library no traia InChIKey, usamos la canonical como principal
    if 'InChIKey_rdkit' in smiles_df.columns:
        smiles_df['InChIKey'] = smiles_df['InChIKey_rdkit']
        print(f"  (Library sin InChIKey nativo; uso RDKit canonical como InChIKey principal)")
    else:
        print(f"  ERROR: no hay InChIKey nativo ni RDKit disponible.")
        sys.exit(1)
else:
    print(f"  InChIKey nativo del library mantenido: {smiles_df['InChIKey'].notna().sum()}/{len(smiles_df)}")

# ---------------------------------------------------------------------
# 4. Cargar MoA annotation (best effort, no es bloqueante)
# ---------------------------------------------------------------------
print(f"\n{'-'*70}\n4. Cargar MoA library\n{'-'*70}")
moa_df = None
if MOA_FILE.exists():
    try:
        moa_df = pd.read_csv(MOA_FILE, encoding='latin-1')
    except UnicodeDecodeError:
        moa_df = pd.read_csv(MOA_FILE, encoding='utf-8', errors='replace')
    # Normalizar columna EOS (case-insensitive)
    moa_eos_col = find_col(moa_df, ['EOS_id', 'EOS', 'eos', 'eos_id', 'eosid', 'compound_id'])
    if moa_eos_col and moa_eos_col != 'EOS_id':
        moa_df = moa_df.rename(columns={moa_eos_col: 'EOS_id'})
    print(f"  MoA library: {len(moa_df)} compuestos, {moa_df.shape[1]} cols")
else:
    print(f"  AVISO: {MOA_FILE} no encontrado. Continuamos sin MoA library.")

# ---------------------------------------------------------------------
# 5. Marcar hits 3D
# ---------------------------------------------------------------------
print(f"\n{'-'*70}\n5. Marcar hits 3D\n{'-'*70}")
hits_set = set()
if HITS_FILE.exists():
    hits = pd.read_csv(HITS_FILE)
    hits_set = set(hits['EOS_id'].dropna().unique())
    print(f"  Hits 3D: {len(hits_set)}")
else:
    print(f"  AVISO: {HITS_FILE} no encontrado.")

# ---------------------------------------------------------------------
# 6. Merge final
# ---------------------------------------------------------------------
print(f"\n{'-'*70}\n6. Merge final\n{'-'*70}")
# Quedarnos con cols esenciales del consensus (no perfiles enteros, solo metadata)
meta_cols = ['EOS_id', 'Source_plate', 'Metadata_Plate', 'Metadata_Replicates_used',
             'Metadata_n_replicates']
meta_cols = [c for c in meta_cols if c in consensus.columns]
consensus_meta = consensus[meta_cols].drop_duplicates(subset=['EOS_id'])

# Una entrada por compuesto (priorizando placa con mas replicates)
if 'Metadata_n_replicates' in consensus_meta.columns:
    consensus_meta = (consensus_meta
                      .sort_values('Metadata_n_replicates', ascending=False)
                      .drop_duplicates(subset=['EOS_id'], keep='first'))

# Merge con SMILES (incluyendo InChIKey_rdkit canonical si existe)
keep_smiles = ['EOS_id', 'SMILES', 'InChIKey', 'InChIKey_rdkit']
keep_smiles = [c for c in keep_smiles if c in smiles_df.columns]
annotated = consensus_meta.merge(smiles_df[keep_smiles], on='EOS_id', how='left')

# Merge con MoA si disponible
if moa_df is not None and 'EOS_id' in moa_df.columns:
    # Mapping de columnas EUopen_* -> nombres estandar (igual que script 05)
    moa_rename = {
        'EUopen_name': 'Drug_name',
        'EUopen_target_name': 'Target_name',
        'EUopen_gene_name': 'Gene_name',
        'EUopen_moa': 'MoA',
        'EUopen_target_type': 'Target_type',
        'EUopen_no. targets': 'N_targets',
        'EUopen_mw': 'Molecular_weight',
        'EUopen_cas': 'CAS',
        'EUopen_synonyms': 'Synonyms',
        'EUopen_GTOPDB [LEVEL 1]': 'GTOPDB_L1',
        'EUopen_GTOPDB [LEVEL 2]': 'GTOPDB_L2',
        'EUopen_GTOPDB [LEVEL 3]': 'GTOPDB_L3',
        'EUopen_GTOPDB [LEVEL 4]': 'GTOPDB_L4',
        'EUopen_ChEMBL [LEVEL 1]': 'ChEMBL_L1',
        'EUopen_ChEMBL [LEVEL 2]': 'ChEMBL_L2',
        'EUopen_ChEMBL [LEVEL 3]': 'ChEMBL_L3',
        'EUopen_ChEMBL [LEVEL 4]': 'ChEMBL_L4',
        'EUopen_Reactome [LEVEL 1]': 'Reactome_L1',
        'EUopen_Reactome [LEVEL 2]': 'Reactome_L2',
        'EUopen_Reactome [LEVEL 3]': 'Reactome_L3',
        'EUopen_inchikey': 'InChIKey_moa',
    }
    moa_df = moa_df.rename(columns={k: v for k, v in moa_rename.items()
                                    if k in moa_df.columns})
    # Quedarnos solo con las columnas mapeadas + EOS_id
    keep_moa = ['EOS_id'] + [v for v in moa_rename.values() if v in moa_df.columns]
    annotated = annotated.merge(moa_df[keep_moa].drop_duplicates(subset=['EOS_id']),
                                on='EOS_id', how='left')
    print(f"  Anotacion MoA mergeada. Columnas anotadas: {[c for c in keep_moa if c != 'EOS_id']}")

# Flag hit 3D
annotated['is_hit_3D'] = annotated['EOS_id'].isin(hits_set)

# ---------------------------------------------------------------------
# 7. Auditoria
# ---------------------------------------------------------------------
print(f"\n{'-'*70}\n7. Auditoria\n{'-'*70}")
n_total = len(annotated)
n_smiles = annotated['SMILES'].notna().sum() if 'SMILES' in annotated.columns else 0
n_inchikey = annotated['InChIKey'].notna().sum() if 'InChIKey' in annotated.columns else 0
n_inchikey_rdkit = annotated['InChIKey_rdkit'].notna().sum() if 'InChIKey_rdkit' in annotated.columns else 0
n_inchikey_moa = annotated['InChIKey_moa'].notna().sum() if 'InChIKey_moa' in annotated.columns else 0
n_drug = annotated['Drug_name'].notna().sum() if 'Drug_name' in annotated.columns else 0
n_moa = annotated['MoA'].notna().sum() if 'MoA' in annotated.columns else 0
n_hits = int(annotated['is_hit_3D'].sum())

# Disagreement entre las 3 columnas InChIKey (diagnostico util)
disagree_count = 0
if all(c in annotated.columns for c in ['InChIKey', 'InChIKey_rdkit', 'InChIKey_moa']):
    sub = annotated.dropna(subset=['InChIKey', 'InChIKey_rdkit', 'InChIKey_moa'])
    disagree = (sub['InChIKey'] != sub['InChIKey_rdkit']) | (sub['InChIKey'] != sub['InChIKey_moa'])
    disagree_count = int(disagree.sum())

print(f"  Total compuestos:         {n_total}")
print(f"  Con SMILES:               {n_smiles} ({100*n_smiles/n_total:.1f}%)")
print(f"  Con InChIKey (nativo):    {n_inchikey} ({100*n_inchikey/n_total:.1f}%)")
print(f"  Con InChIKey_rdkit:       {n_inchikey_rdkit} ({100*n_inchikey_rdkit/n_total:.1f}%)")
print(f"  Con InChIKey_moa:         {n_inchikey_moa} ({100*n_inchikey_moa/n_total:.1f}%)")
print(f"  Discrepancias entre IKs:  {disagree_count} (estos son los compuestos donde el matching JUMP necesitara las 3 columnas)")
print(f"  Con Drug_name:            {n_drug} ({100*n_drug/n_total:.1f}%)")
print(f"  Con MoA documentado:      {n_moa} ({100*n_moa/n_total:.1f}%)")
print(f"  Hits 3D marcados:         {n_hits}")

# ---------------------------------------------------------------------
# 8. Guardar
# ---------------------------------------------------------------------
out_csv = OUT_DIR / 'cp3d_library_annotated.csv'
annotated.to_csv(out_csv, index=False)
print(f"\n  Guardado: {out_csv}")

audit_path = OUT_DIR / 'cp3d_library_annotated_audit.txt'
with open(audit_path, 'w', encoding='utf-8') as f:
    f.write("=== CP3D LIBRARY ANNOTATED — AUDIT ===\n\n")
    f.write(f"Total compuestos:         {n_total}\n")
    f.write(f"Con SMILES:               {n_smiles} ({100*n_smiles/n_total:.1f}%)\n")
    f.write(f"Con InChIKey:             {n_inchikey} ({100*n_inchikey/n_total:.1f}%)\n")
    f.write(f"Con Drug_name:            {n_drug} ({100*n_drug/n_total:.1f}%)\n")
    f.write(f"Con MoA:                  {n_moa} ({100*n_moa/n_total:.1f}%)\n")
    f.write(f"Hits 3D marcados:         {n_hits}\n\n")
    f.write("Compuestos sin SMILES (top 20):\n")
    missing = annotated.loc[annotated.get('SMILES', pd.Series()).isna(), 'EOS_id'].head(20)
    for x in missing:
        f.write(f"  {x}\n")
print(f"  Auditoria: {audit_path}")

print(f"\n{'='*70}\nDONE. Siguiente paso: ejecutar 11_jump_cp_overlap.py\n{'='*70}\n")
