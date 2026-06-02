"""
run_full_pipeline.py (v5 — adds STAGE 9: multivariate analysis)
=====================================================================
Script maestro que ejecuta el pipeline completo desde plate maps hasta
analisis multivariante (PCA / t-SNE / UMAP) de los hits.

NUEVO en v5:
- STAGE 9: 09_multivariate_analysis.py (PCA + t-SNE + UMAP + tests
  de proximidad y validacion fenotipica; reproduce seccion 4.7 del informe)
- --target_group / --target_label / --n_permutations: parametros del
  analisis multivariante; valores por defecto = los 3 inhibidores HSP90

NUEVO en v4:
- --start_from N: salta etapas anteriores (no limpia ni regenera lo previo)
- --min_replicates N: pasa el filtro a script 05 (default 3)

NO borra:
- data\\raw           (CSVs originales de CellProfiler)
- C:\\Users\\Ianezi\\Documents\\CP3D\\Platemaps\\  (plate maps source)
- scripts\\           (los scripts del pipeline)

Borra y regenera (con --skip_clean=False, default):
- data\\processed\\
- data\\platemaps\\processed\\
- results\\           (todas las subcarpetas, incluyendo integrated_hits)

Etapas:
1. process_platemaps + process_platemap_C2388
2. split_C2388_replicates (si C2388R1-R4 no existen)
3. Por cada placa-replica: 01_merge_and_qc + 02_normalize
4. Por cada placa: 03_combine_and_score + 04_robust_analysis
5. Por cada placa: 05_merge_hits_chemistry  [aplica --min_replicates]
6. Por cada placa: 06_compare_hits_vs_library
7. Analisis combinado de las 3 placas: 07_combined_analysis
8. Visualizacion integrada de todos los hits: 08_visualize_all_hits
9. Analisis multivariante PCA/t-SNE/UMAP: 09_multivariate_analysis

Uso:
    # Pipeline completo desde cero
    python run_full_pipeline.py
    
    # Solo desde etapa 5 (para reaplicar filtro tras cambiar min_replicates)
    python run_full_pipeline.py --start_from 5 --min_replicates 3
    
    # Solo etapa 9 (rehacer multivariate sin tocar lo demas)
    python run_full_pipeline.py --start_from 9
    
    # Multivariate con otro target group
    python run_full_pipeline.py --start_from 9 \\
        --target_group EOS101430 EOS101302 \\
        --target_label expanded_hits
    
    # Solo placas seleccionadas
    python run_full_pipeline.py --plates C2386 C2388
    
    # Saltar limpieza
    python run_full_pipeline.py --skip_clean
"""

import argparse
import sys
import shutil
import subprocess
from pathlib import Path

BASE = Path(r'C:\Users\Ianezi\Documents\CP3D\analysis')
SCRIPTS_DIR = BASE / 'scripts'
RAW_DIR = BASE / 'data' / 'raw'

DEFAULT_PLATES = ['C2386', 'C2387', 'C2388']
MULTI_REP_PLATES = {'C2388'}
REPLICATES = ['R1', 'R2', 'R3', 'R4']

FOLDERS_TO_CLEAN = [
    BASE / 'data' / 'processed',
    BASE / 'data' / 'platemaps' / 'processed',
    BASE / 'results' / 'qc',
    BASE / 'results' / 'norm',
    BASE / 'results' / 'multi_rep',
    BASE / 'results' / 'enrichment',
    BASE / 'results' / 'hits_summary',
    BASE / 'results' / 'integrated_hits',
    BASE / 'results' / 'multivariate',
]

# Hit calling parameters per plate
HIT_PARAMS_03 = {
    'C2386': [],
    'C2387': [],
    'C2388': ['--hit_threshold', '80'],
}
HIT_PARAMS_04 = {
    'C2386': [],
    'C2387': [],
    'C2388': ['--activity_threshold_pct', '70', '--final_hit_pct', '80'],
}

# Multivariate analysis defaults (STAGE 9)
DEFAULT_TARGET_GROUP = ['EOS101988', 'EOS100193', 'EOS100198']  # 3 HSP90 inhibitors
DEFAULT_TARGET_LABEL = 'HSP90'
DEFAULT_N_PERMUTATIONS = 10000


def log(msg, level=1):
    prefix = "  " * (level - 1)
    print(f"{prefix}{msg}", flush=True)


def section(title, char='='):
    log("")
    log(char * 70)
    log(f"=== {title} ===")
    log(char * 70)


def run_cmd(cmd, label=None):
    if label:
        log(f"\n>>> {label}", level=1)
    log(f"  Command: {' '.join(cmd)}", level=2)
    result = subprocess.run(cmd, cwd=SCRIPTS_DIR, capture_output=False)
    if result.returncode != 0:
        log(f"  *** FAILED with exit code {result.returncode} ***", level=2)
        return False
    return True


def clean_folders():
    section("STEP 0: Clean previous outputs")
    for folder in FOLDERS_TO_CLEAN:
        if folder.exists():
            log(f"Removing: {folder}", level=2)
            try:
                shutil.rmtree(folder)
            except Exception as e:
                log(f"  WARNING: could not remove {folder}: {e}", level=2)
        else:
            log(f"Skipping (does not exist): {folder}", level=2)
    for folder in FOLDERS_TO_CLEAN:
        folder.mkdir(parents=True, exist_ok=True)
    log("All output folders cleaned and recreated.", level=1)


def stage_platemaps(plates):
    section("STAGE 1: Process plate maps")
    log("\n>>> process_platemaps.py", level=1)
    if not run_cmd([sys.executable, 'process_platemaps.py']):
        return False
    for plate in plates:
        if plate in MULTI_REP_PLATES:
            script_name = f'process_platemap_{plate}.py'
            if not (SCRIPTS_DIR / script_name).exists():
                log(f"  WARNING: {script_name} not found", level=2)
                continue
            log(f"\n>>> {script_name}", level=1)
            if not run_cmd([sys.executable, script_name]):
                return False
    return True


def stage_split_multirep(plates):
    section("STAGE 2: Split multi-replicate plates")
    for plate in plates:
        if plate not in MULTI_REP_PLATES:
            continue
        # Check if all replicate folders exist with files
        all_ok = all(
            (RAW_DIR / f'{plate}{rep}').exists() and 
            any((RAW_DIR / f'{plate}{rep}').iterdir())
            for rep in REPLICATES
        )
        if all_ok:
            log(f"\n>>> {plate}: replicate folders already exist, skipping split", level=1)
            continue
        script_name = f'split_{plate}_replicates.py'
        if not (SCRIPTS_DIR / script_name).exists():
            log(f"  WARNING: {script_name} not found", level=2)
            continue
        log(f"\n>>> {script_name}", level=1)
        if not run_cmd([sys.executable, script_name]):
            return False
    return True


def stage_per_replicate(plates):
    section("STAGE 3: Process each replicate (01 + 02)")
    for plate in plates:
        log(f"\n--- Plate {plate} ---", level=1)
        for rep in REPLICATES:
            folder = f'{plate}{rep}'
            raw_folder = RAW_DIR / folder
            if not raw_folder.exists() or not any(raw_folder.iterdir()):
                log(f"  Skipping {folder} (raw folder empty/missing)", level=2)
                continue
            log(f"\n>>> {folder} - 01_merge_and_qc", level=1)
            if not run_cmd([sys.executable, '01_merge_and_qc.py', '--plate', plate, '--folder', folder]):
                continue
            log(f"\n>>> {folder} - 02_normalize", level=1)
            if not run_cmd([sys.executable, '02_normalize.py', '--folder', folder]):
                continue
    return True


def stage_multi_replicate(plates):
    section("STAGE 4: Multi-replicate analysis (03 + 04)")
    for plate in plates:
        log(f"\n--- Plate {plate} ---", level=1)
        rep_folders = []
        for rep in REPLICATES:
            normalized_csv = BASE / 'data' / 'processed' / f'{plate}{rep}_normalized_dmso_selected.csv'
            if normalized_csv.exists():
                rep_folders.append(f'{plate}{rep}')
        if len(rep_folders) < 2:
            log(f"  Skipping {plate}: only {len(rep_folders)} replicate(s)", level=2)
            continue
        log(f"  Replicates available: {rep_folders}", level=2)
        # 03
        log(f"\n>>> {plate} - 03_combine_and_score", level=1)
        cmd = [sys.executable, '03_combine_and_score.py', '--plate', plate, '--replicates'] + rep_folders + HIT_PARAMS_03.get(plate, [])
        if not run_cmd(cmd):
            continue
        # 04
        log(f"\n>>> {plate} - 04_robust_analysis", level=1)
        cmd = [sys.executable, '04_robust_analysis.py', '--plate', plate, '--replicates'] + rep_folders + HIT_PARAMS_04.get(plate, [])
        if not run_cmd(cmd):
            continue
    return True


def stage_chemistry(plates, min_replicates=3):
    section(f"STAGE 5: Merge hits with chemistry/MoA (05) — min_replicates={min_replicates}")
    for plate in plates:
        confirmed_hits = BASE / 'data' / 'processed' / f'confirmed_hits_{plate}.csv'
        if not confirmed_hits.exists():
            log(f"  Skipping {plate} (no confirmed_hits CSV)", level=2)
            continue
        log(f"\n>>> {plate} - 05_merge_hits_chemistry", level=1)
        run_cmd([sys.executable, '05_merge_hits_chemistry.py',
                 '--plate', plate,
                 '--min_replicates', str(min_replicates)])
    return True


def stage_enrichment_per_plate(plates, min_replicates=3):
    section(f"STAGE 6: Hits vs library enrichment per plate (06) — min_replicates={min_replicates}")
    if not (SCRIPTS_DIR / '06_compare_hits_vs_library.py').exists():
        log(f"  Skipping: 06 not found", level=2)
        return True
    for plate in plates:
        confirmed_hits = BASE / 'data' / 'processed' / f'confirmed_hits_{plate}.csv'
        if not confirmed_hits.exists():
            log(f"  Skipping {plate}", level=2)
            continue
        log(f"\n>>> {plate} - 06_compare_hits_vs_library", level=1)
        run_cmd([sys.executable, '06_compare_hits_vs_library.py',
                 '--plate', plate,
                 '--min_replicates', str(min_replicates)])
    return True


def stage_combined(plates, min_replicates=3):
    section(f"STAGE 7: Combined analysis across all plates (07) — min_replicates={min_replicates}")
    if not (SCRIPTS_DIR / '07_combined_analysis.py').exists():
        log(f"  Skipping: 07 not found", level=2)
        return True
    log(f"\n>>> 07_combined_analysis", level=1)
    cmd = [sys.executable, '07_combined_analysis.py',
           '--plates'] + plates + ['--min_replicates', str(min_replicates)]
    run_cmd(cmd)
    return True


def stage_visualize(plates):
    section("STAGE 8: Integrated visualization of all hits (08)")
    if not (SCRIPTS_DIR / '08_visualize_all_hits.py').exists():
        log(f"  Skipping: 08 not found", level=2)
        return True
    # Verify there are hits to visualize
    has_hits = any(
        (BASE / 'results' / 'hits_summary' / f'{p}_hits_with_chemistry.csv').exists()
        for p in plates
    )
    if not has_hits:
        log(f"  Skipping: no hits_with_chemistry CSV files found", level=2)
        return True
    log(f"\n>>> 08_visualize_all_hits", level=1)
    cmd = [sys.executable, '08_visualize_all_hits.py']
    run_cmd(cmd)
    return True


def stage_multivariate(target_group, target_label, n_permutations):
    section(f"STAGE 9: Multivariate analysis PCA/t-SNE/UMAP (09) - target={target_label}")
    if not (SCRIPTS_DIR / '09_multivariate_analysis.py').exists():
        log(f"  Skipping: 09 not found", level=2)
        return True
    # Verify required inputs exist (consensus + hits with chemistry)
    has_consensus = all(
        (BASE / 'data' / 'processed' / f'consensus_{p}.csv').exists()
        for p in DEFAULT_PLATES
    )
    has_hits = any(
        (BASE / 'results' / 'hits_summary' / f'{p}_hits_with_chemistry.csv').exists()
        for p in DEFAULT_PLATES
    )
    if not has_consensus:
        log(f"  Skipping: consensus_<plate>.csv files missing (need to run stages 4)", level=2)
        return True
    if not has_hits:
        log(f"  Skipping: no hits_with_chemistry CSV files found (need to run stage 5)", level=2)
        return True
    log(f"\n>>> 09_multivariate_analysis (target={target_label}, n_perm={n_permutations})", level=1)
    cmd = [sys.executable, '09_multivariate_analysis.py',
           '--target_group'] + list(target_group) + [
           '--target_label', target_label,
           '--n_permutations', str(n_permutations)]
    run_cmd(cmd)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plates', nargs='+', default=DEFAULT_PLATES)
    parser.add_argument('--skip_clean', action='store_true',
                        help='Skip cleanup step (preserves existing outputs)')
    parser.add_argument('--start_from', type=int, default=1, choices=range(1, 10),
                        help='Start from stage N (1-9). Skips earlier stages and cleanup.')
    parser.add_argument('--min_replicates', type=int, default=3,
                        help='Minimum N_replicates filter for hits in stage 5 (default: 3)')
    parser.add_argument('--target_group', nargs='+', default=DEFAULT_TARGET_GROUP,
                        help='EOS_ids of the target group for stage 9 multivariate analysis '
                             '(default: 3 HSP90 inhibitors)')
    parser.add_argument('--target_label', default=DEFAULT_TARGET_LABEL,
                        help='Short label for the target group (default: HSP90)')
    parser.add_argument('--n_permutations', type=int, default=DEFAULT_N_PERMUTATIONS,
                        help='Permutations for proximity tests in stage 9 (default: 10000)')
    args = parser.parse_args()
    plates = args.plates
    start_stage = args.start_from
    min_reps = args.min_replicates
    target_group = args.target_group
    target_label = args.target_label
    n_perm = args.n_permutations

    section(f"FULL PIPELINE: {plates}", char='#')
    log(f"Base directory:     {BASE}")
    log(f"Plates:             {plates}")
    log(f"Multi-replicate:    {[p for p in plates if p in MULTI_REP_PLATES]}")
    log(f"Start from stage:   {start_stage}")
    log(f"Min replicates:     {min_reps}")
    log(f"Target group (S9):  {target_label} = {target_group}")
    log(f"N permutations (S9):{n_perm}")
    
    # Cleanup only if starting from stage 1 and not skipped
    if start_stage == 1 and not args.skip_clean:
        clean_folders()
    elif start_stage > 1:
        log(f"\nSkipping cleanup (starting from stage {start_stage})", level=1)
    
    if start_stage <= 1:
        if not stage_platemaps(plates):
            log("\nABORTED at platemaps stage")
            sys.exit(1)
    
    if start_stage <= 2:
        if not stage_split_multirep(plates):
            log("\nABORTED at split stage")
            sys.exit(1)
    
    if start_stage <= 3:
        stage_per_replicate(plates)
    
    if start_stage <= 4:
        stage_multi_replicate(plates)
    
    if start_stage <= 5:
        stage_chemistry(plates, min_replicates=min_reps)
    
    if start_stage <= 6:
        stage_enrichment_per_plate(plates, min_replicates=min_reps)
    
    if start_stage <= 7:
        stage_combined(plates, min_replicates=min_reps)
    
    if start_stage <= 8:
        stage_visualize(plates)

    if start_stage <= 9:
        stage_multivariate(target_group, target_label, n_perm)

    section("PIPELINE COMPLETE", char='#')
    log("Outputs in:", level=1)
    log("  data\\platemaps\\processed\\         platemaps procesados", level=2)
    log("  data\\processed\\                   CSVs intermedios", level=2)
    log("  results\\qc\\                       QC plots por replica", level=2)
    log("  results\\norm\\                     normalization QC", level=2)
    log("  results\\multi_rep\\                analisis multi-replica", level=2)
    log("  results\\hits_summary\\             hits con quimica/MoA", level=2)
    log("  results\\enrichment\\               enriquecimiento por placa", level=2)
    log("  results\\enrichment\\combined\\      analisis combinado 3 placas", level=2)
    log("  results\\integrated_hits\\          visualizaciones de los hits", level=2)
    log("  results\\multivariate\\             PCA/t-SNE/UMAP + tests proximidad", level=2)


if __name__ == '__main__':
    main()