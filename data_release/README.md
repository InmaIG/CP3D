# Data release — CP3D paper (Communications Biology)

Processed per-compound data supporting Iañez et al., "3D Cell Painting in HepG2 spheroids"
(Commun Biol, 2026).

## Files

- **per_compound.csv** — Master table, one row per compound (n = 735). Columns:
  `EOS_id`, `Drug_name`, `n_features_active` (2D active feature count at |z| > 3),
  `activity_RMS_2D`, `activity_RMS_3D`, `rank_2D_n_active_pct`, `rank_2D_RMS_pct`,
  `rank_3D_RMS_pct`, `is_hit_3D`.
- **per_hit_3D_in_2D.csv** — The 25 confirmed 3D hits with their paired 2D activity
  metrics and within-library ranks.

## Regeneration

Both files are produced by `analysis/scripts/11_compare_2d_medina_vs_3d.py`, run after
`run_full_pipeline.py` and `10_build_annotated_library.py`. See the root README for the
full pipeline.

## Citation

If you use these data, please cite:
Iañez I, *et al.* *3D Cell Painting in HepG2 spheroids for phenotypic drug discovery.*
Commun Biol (2026).