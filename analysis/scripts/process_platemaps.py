"""
process_platemaps.py
====================
Procesa los plate maps de C2386, C2387, C2388:
- Lee los Excel originales
- Convierte nomenclatura A1 -> A01 (zero-padded, formato CellProfiler)
- Anade los wells de cols 23 y 24 como controles
- Anade columna Plate y Concentration
- Guarda CSVs procesados (uno por placa + uno combinado)

Uso:
    python process_platemaps.py
"""

import pandas as pd
import re
from pathlib import Path

# ---------------- Configuracion ----------------
INPUT_DIR = Path(r'C:\Users\Ianezi\Documents\CP3D\Platemaps')
OUTPUT_DIR = Path(r'C:\Users\Ianezi\Documents\CP3D\analysis\data\platemaps\processed')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PLATES = ['C2386', 'C2387', 'C2388']

# Layout estandar 384w
ROWS = list('ABCDEFGHIJKLMNOP')   # 16 filas
COLS = list(range(1, 25))          # 24 columnas

# Configuracion de controles (CONFIRMA con quien preparo la placa)
CONTROL_COLS = {
    23: 'DMSO',         # Vehicle control
    24: 'DMSO',   # Positive toxicity control
}

# Concentracion de los compuestos del library set (segun protocolo del proyecto)
COMPOUND_CONCENTRATION_uM = 10.0
DMSO_CONCENTRATION_uM = 0.0
NOCODAZOLE_CONCENTRATION_uM = 10.0  # Ajusta si usas otra concentracion


def normalize_well(well_str):
    """Convert 'A1' -> 'A01', 'P22' -> 'P22'."""
    m = re.match(r'^([A-Z]+)(\d+)$', str(well_str).strip())
    if not m:
        raise ValueError(f"Well no reconocido: {well_str}")
    row, col = m.group(1), int(m.group(2))
    return f"{row}{col:02d}"


def process_plate(plate_name):
    """Process one platemap: read, normalize wells, add controls, add metadata."""
    fn = INPUT_DIR / f'Platemap_{plate_name}.xlsx'
    print(f"\n--- Procesando {plate_name} ---")
    
    if not fn.exists():
        print(f"  ATENCION: No se encuentra {fn}")
        return None
    
    df = pd.read_excel(fn)
    print(f"  Wells originales: {len(df)}")
    
    # Rename column for clarity
    df = df.rename(columns={'Molecule name': 'Compound'})
    
    # Normalize well naming: A1 -> A01
    df['Well'] = df['Well'].apply(normalize_well)
    
    # Add metadata
    df['Plate'] = plate_name
    df['Concentration_uM'] = COMPOUND_CONCENTRATION_uM
    df['Well_type'] = 'compound'
    
    # Build list of controls (wells in CONTROL_COLS that aren't already in plate map)
    existing_wells = set(df['Well'])
    control_rows = []
    for col_num, ctrl_name in CONTROL_COLS.items():
        for row in ROWS:
            well = f"{row}{col_num:02d}"
            if well not in existing_wells:
                conc = (DMSO_CONCENTRATION_uM if ctrl_name == 'DMSO' 
                        else NOCODAZOLE_CONCENTRATION_uM)
                control_rows.append({
                    'Well': well,
                    'Compound': ctrl_name,
                    'Plate': plate_name,
                    'Concentration_uM': conc,
                    'Well_type': 'control',
                })
    
    if control_rows:
        controls_df = pd.DataFrame(control_rows)
        df = pd.concat([df, controls_df], ignore_index=True)
        print(f"  Wells de control anadidos: {len(controls_df)}")
        print(f"    DMSO: {(controls_df['Compound']=='DMSO').sum()}")
        print(f"    Nocodazole: {(controls_df['Compound']=='Nocodazole').sum()}")
    
    # Sort by Well for readability
    df['_row_letter'] = df['Well'].str[0]
    df['_col_num'] = df['Well'].str[1:].astype(int)
    df = df.sort_values(['_col_num', '_row_letter']).reset_index(drop=True)
    df = df.drop(columns=['_row_letter', '_col_num'])
    
    # Reorder columns
    df = df[['Plate', 'Well', 'Compound', 'Concentration_uM', 'Well_type']]
    
    print(f"  Wells totales tras anadir controles: {len(df)}")
    print(f"  Compuestos unicos: {df['Compound'].nunique()}")
    
    return df


def main():
    all_dfs = []
    
    for plate in PLATES:
        df = process_plate(plate)
        if df is None:
            continue
        
        # Save individual platemap
        out_csv = OUTPUT_DIR / f'platemap_{plate}_processed.csv'
        df.to_csv(out_csv, index=False)
        print(f"  Guardado: {out_csv}")
        
        all_dfs.append(df)
    
    # Save combined platemap
    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        out_combined = OUTPUT_DIR / 'platemap_all_combined.csv'
        combined.to_csv(out_combined, index=False)
        print(f"\n--- Resumen combinado ---")
        print(f"Total filas: {len(combined)}")
        print(f"Placas: {combined['Plate'].unique()}")
        print(f"Tipos de well: {combined['Well_type'].value_counts().to_dict()}")
        print(f"Wells por placa:")
        print(combined.groupby('Plate')['Well_type'].value_counts())
        print(f"\nGuardado: {out_combined}")
        
        print(f"\n--- Sample del archivo combinado ---")
        print(combined.head(10).to_string(index=False))


if __name__ == '__main__':
    main()