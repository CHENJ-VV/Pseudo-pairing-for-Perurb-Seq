from pathlib import Path
import sys

# If this notebook is copied together with the .py files, set WORKFLOW_DIR to that folder.
# On IBEX, I recommend copying the whole folder to something like:
# /ibex/user/chenj0i/Perturbation/scripts/pseudo_pairing_refactor
WORKFLOW_DIR = Path("/ibex/user/chenj0i/Perturbation/SEACells/Junfan_scripts/Replogle_RPE/pseudo_pairing_refactor")

# Example explicit path if you copy the folder to your scripts directory:
# WORKFLOW_DIR = Path('/ibex/user/chenj0i/Perturbation/scripts/pseudo_pairing_refactor')

sys.path.insert(0, str(WORKFLOW_DIR))
print('Workflow dir:', WORKFLOW_DIR)

from run_pseudo_pairing_repetitions import run_pseudo_pairing_repetition_plan
from pseudo_pairing_utils import STRATEGY_ORDER

print('Available strategy order:')
for s in STRATEGY_ORDER:
    print(' -', s)

# ============================================================
# Dataset paths
# ============================================================
DATASET_ID = 'Replogle_RPE'

PROCESSED_ROOT = Path('/ibex/user/chenj0i/Perturbation/data/processed_data') / DATASET_ID
GROUP_ROOT = PROCESSED_ROOT / 'groups'

CONTROL_H5AD = GROUP_ROOT / f'{DATASET_ID}_control_processed.h5ad'

# Keep or remove groups depending on which files exist after preprocessing.
PERTURBED_H5ADS = {
    'single': GROUP_ROOT / f'{DATASET_ID}_single_processed.h5ad',
    # 'dual': GROUP_ROOT / f'{DATASET_ID}_dual_processed.h5ad',
    # 'multi': GROUP_ROOT / f'{DATASET_ID}_multi_processed.h5ad',
}

OUTDIR = Path('/ibex/project/c2366/Perturb_data/Replogle_rpe_data')

print('Control:', CONTROL_H5AD)
print('Perturbed groups:')
for k, v in PERTURBED_H5ADS.items():
    print(f'  {k}: {v}')
print('Output root:', OUTDIR / DATASET_ID)


# ============================================================
# Strategy selection
# ============================================================
STRATEGIES_TO_RUN = [
    # 'S0_naive_mean_control_reference',
    # 'S1_random_single_control',
    # 'S2_random_average_controls',
    # 'S3_SEACell_metacell_average',
    # 'S4_SEACell_balanced_random_sample',
    'S5_SEACell_OT_sampled_average',
]

# ============================================================
# Repetition settings
# ============================================================
RANDOM_SEEDS = [0, 1, 2, 3, 4]

# Debug option. Use 300 for fast testing, None for the full dataset.
MAX_PAIRS_PER_PERTURBATION = None
PAIR_SELECTION_SEED = 42

# ============================================================
# Random baseline parameters
# ============================================================
S2_N_CONTROL_CELLS_TO_AVERAGE_VALUES = [100]
SAMPLING_REPLACE = True

# ============================================================
# SEACell settings reused by S3/S4/S5
# ============================================================
SEACELL_SETTINGS = [
    # {
    # 'setting_id': 'nmc_50',
    # 'n_metacells': 50,
    # 'n_waypoint_eigs': 10,
    # 'seacells_n_iter': 100,
    # 'seacell_seed': 42,
    # },
    # {
    #     'setting_id': 'nmc_100',
    #     'n_metacells': 100,
    #     'n_waypoint_eigs': 10,
    #     'seacells_n_iter': 100,
    #     'seacell_seed': 42,
    # },
    # {
    #     'setting_id': 'nmc_200',
    #     'n_metacells': 200,
    #     'n_waypoint_eigs': 10,
    #     'seacells_n_iter': 100,
    #     'seacell_seed': 42,
    # },
    # {
    #     'setting_id': 'nmc_350',
    #     'n_metacells': 350,
    #     'n_waypoint_eigs': 10,
    #     'seacells_n_iter': 100,
    #     'seacell_seed': 42,
    # },
    {
        'setting_id': 'nmc_500',
        'n_metacells': 500,
        'n_waypoint_eigs': 10,
        'seacells_n_iter': 100,
        'seacell_seed': 42,
    },
]

# S3: randomly sample k metacells and average their metacell expression profiles.
S3_N_METACELLS_TO_AVERAGE_VALUES = [3, 5, 10]
SAMPLE_METACELLS_WITH_REPLACEMENT = True

# S5: OT top-k metacells, then sample true control cells from each matched metacell.
# S5_TOP_K_VALUES = [1, 3, 5]
# S5_TOP_K_VALUES = [5]
S5_TOP_K_VALUES = [5]
SAMPLE_CELLS_PER_METACELL = 10
OT_REG = 0.05
CONTROL_MASS = 'size'  # 'size' or 'uniform'

# ============================================================
# Data keys
# ============================================================
EXPR_LAYER = 'X'
EMBEDDING_KEY = 'X_pca'

# Use 'auto' to infer from obs columns.  For your processed files, likely candidates are:
# Replogle: 'gene'
# Norman/scPerturb-style: 'condition' or the standardized 'perturbation_key'
PERTURBATION_KEY = 'auto'

CONFIG = {
    'dataset_id': DATASET_ID,
    'control_h5ad': str(CONTROL_H5AD),
    'perturbed_h5ads': {k: str(v) for k, v in PERTURBED_H5ADS.items()},
    'outdir': str(OUTDIR),

    # Which strategies to run.
    'strategies_to_run': STRATEGIES_TO_RUN,

    # Data keys.
    'expr_layer': EXPR_LAYER,
    'embedding_key': EMBEDDING_KEY,
    'perturbation_key': PERTURBATION_KEY,
    'require_all_genes': False,

    # Repeats and debug subset.
    'random_seeds': RANDOM_SEEDS,
    'max_pairs_per_perturbation': MAX_PAIRS_PER_PERTURBATION,
    'pair_selection_seed': PAIR_SELECTION_SEED,

    # General matrix construction.
    'batch_size': 4096,
    'matrix_batch_size': 4096,
    'sampling_replace': SAMPLING_REPLACE,
    'store_sampled_control_positions': True,

    # Random-average baseline.
    's2_n_control_cells_to_average_values': S2_N_CONTROL_CELLS_TO_AVERAGE_VALUES,

    # SEACell membership settings.
    'membership_root': str(OUTDIR / DATASET_ID / '_seacell_memberships'),
    'seacell_settings': SEACELL_SETTINGS,
    'seacell_key_prefix': 'SEACell',
    'n_waypoint_eigs': 10,
    'seacells_n_iter': 50,
    'seacell_seed': 42,
    'use_gpu_seacells': False,
    'overwrite_memberships': False,

    # S3 metacell-average baseline.
    's3_n_metacells_to_average_values': S3_N_METACELLS_TO_AVERAGE_VALUES,
    'sample_metacells_with_replacement': SAMPLE_METACELLS_WITH_REPLACEMENT,

    # S5 OT settings.
    's5_top_k_values': S5_TOP_K_VALUES,
    'sample_cells_per_metacell': SAMPLE_CELLS_PER_METACELL,
    'ot_reg': OT_REG,
    'ot_max_iter': 2000,
    'ot_tol': 1e-7,
    'cost_metric': 'sqeuclidean',
    'control_mass': CONTROL_MASS,
    'overwrite_assignments': False,
    'overwrite_sampled_outputs': False,
}

missing = []
if not Path(CONFIG['control_h5ad']).exists():
    missing.append(CONFIG['control_h5ad'])
for group, path in CONFIG['perturbed_h5ads'].items():
    if not Path(path).exists():
        missing.append(path)

if missing:
    print('Missing files:')
    for p in missing:
        print(' -', p)
    raise FileNotFoundError('Fix the missing paths above before running the workflow.')
else:
    print('All configured h5ad files exist.')

RUN_PAIRING = True

if RUN_PAIRING:
    manifest = run_pseudo_pairing_repetition_plan(CONFIG)
    # display(manifest)
else:
    print('RUN_PAIRING is False. Config prepared but workflow not launched.')

manifest_path = Path(CONFIG['outdir']) / CONFIG['dataset_id'] / 'pseudo_pairing_repetition_manifest.csv'
if manifest_path.exists():
    manifest = __import__('pandas').read_csv(manifest_path)
    print('Manifest:', manifest_path)
    print('Rows:', manifest.shape[0])
    # display(
    #     manifest.groupby(['perturbed_group', 'strategy'], dropna=False)
    #     .agg(n_datasets=('pseudo_control_h5ad', 'count'))
    #     .reset_index()
    # )
    # display(manifest.head())
else:
    print('Manifest not found yet:', manifest_path)
