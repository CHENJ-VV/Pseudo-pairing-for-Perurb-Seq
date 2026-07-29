# Pseudo-control pairing refactor

This folder contains a dataset-agnostic refactor of the previous Replogle pseudo-control construction notebooks/scripts.

## Files

- `pseudo_pairing_utils.py`  
  Shared IO, AnnData alignment, random sampling, metadata, and save helpers.

- `pseudo_pairing_seacell.py`  
  SEACell membership generation/loading helpers.  Produces and reuses membership files with columns:
  `control_cell_id`, `control_cell_pos`, `metacell_id`.

- `pseudo_pairing_strategies.py`  
  Strategy implementations:
  - `S0_naive_mean_control_reference`
  - `S1_random_single_control`
  - `S2_random_average_controls`
  - `S3_SEACell_metacell_average`
  - `S4_SEACell_balanced_random_sample`
  - `S5_SEACell_OT_sampled_average`

- `run_pseudo_pairing_repetitions.py`  
  The orchestration layer that loops over perturbed groups, strategy order, parameter grids, and repeat seeds.

- `execute_pseudo_pairing_repetitions.ipynb`  
  Jupyter execution notebook.  Edit the config cell and run.

## Basic output layout

For each dataset and perturbed group, outputs are written under:

```text
{OUTDIR}/{DATASET_ID}/{perturbed_group}/{strategy_name}/...
```

The top-level manifest is:

```text
{OUTDIR}/{DATASET_ID}/pseudo_pairing_repetition_manifest.csv
```

Each generated dataset directory contains:

```text
pseudo_control_aligned_to_perturbed.h5ad
pair_metadata.csv or pair_metadata.parquet
pairing_qc_summary.json
strategy_config.json
```

SEACell memberships are stored separately and reused by S3, S4, and S5:

```text
{OUTDIR}/{DATASET_ID}/_seacell_memberships/{setting_id}/membership/control_cell_to_metacell_membership.csv
```

## Notes

1. `perturbation_key='auto'` tries common columns such as `perturbation_key`, `perturbation_label`, `condition`, and `gene`.
2. For a smoke test, set `max_pairs_per_perturbation=300` in the notebook config.
3. For full runs, set `max_pairs_per_perturbation=None`.
4. S5 requires POT (`pip install POT`) and PCA embeddings in both control and perturbed h5ad files under `embedding_key`, usually `X_pca`.
5. S3/S4/S5 require SEACell membership.  If missing, the workflow will attempt to run SEACells using the control h5ad.
