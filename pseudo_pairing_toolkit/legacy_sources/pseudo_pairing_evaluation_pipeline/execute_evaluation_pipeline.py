from pathlib import Path
from types import SimpleNamespace

from run_evaluation_pipeline import run_evaluation_pipeline
from mlp_models import ForwardPerturbationMLP, InversePerturbationMLP

# ============================================================
# User configuration
# ============================================================

DATASET_ID = "Replogle_RPE"
PROCESSED_ROOT = Path("/ibex/user/chenj0i/Perturbation/data/processed_data/Replogle_RPE/groups")
PAIRING_ROOT = Path("/ibex/project/c2366/Perturb_data/Replogle_rpe_data/Replogle_RPE")
EVAL_ROOT = Path("/ibex/user/chenj0i/Perturbation/evaluation/Replogle_RPE_pseudo_pairing_evaluation")

PERTURBED_GROUP = "single"  # change to "dual" or "multi" when evaluating other processed groups

CONFIG = SimpleNamespace(
    dataset_id=DATASET_ID,
    control_h5ad=str(PROCESSED_ROOT / f"{DATASET_ID}_control_processed.h5ad"),
    perturbed_h5ad=str(PROCESSED_ROOT / f"{DATASET_ID}_{PERTURBED_GROUP}_processed.h5ad"),
    manifest_path=str(PAIRING_ROOT / "pseudo_pairing_repetition_manifest.csv"),
    outdir=EVAL_ROOT / PERTURBED_GROUP,
    perturbation_key="perturbation_key",
    perturbed_groups_to_evaluate=[PERTURBED_GROUP],
    strategies_to_evaluate=None,
    evaluation_tasks=["mlp"],
    max_runs_to_evaluate=None,
    max_eval_genes=3000,
    check_shared_genes_across_all_runs=False,
    seed=42,
    # control manifold
    n_pcs=50,
    n_control_sample_for_pca=10000,
    n_control_sample_for_overlap=10000,
    n_pseudo_sample_for_overlap=10000,
    n_mmd_sample=3000,
    source_mixing_k=30,
    # perturbation effect
    min_cells_per_perturbation=20,
    top_de_k_list=[20, 50, 100],
    # MLP
    mlp_tasks=["forward", "inverse_strategy_delta", "inverse_common_delta"],
    device="auto",
    split_seed=42,
    model_seed=42,
    train_frac=0.70,
    val_frac=0.15,
    test_frac=0.15,
    batch_size=256,
    num_workers=0,
    forward_epochs=30,
    inverse_epochs=30,
    early_stop_patience=3,
    learning_rate=1e-3,
    weight_decay=1e-5,
    hidden_dim=1024,
    latent_dim=512,
    pert_emb_dim=256,
    dropout=0.15,
    inverse_topk_accuracy_list=[3, 5, 10],
    topk_effect_genes=[20, 50, 100],
    skip_existing_forward=True,
    skip_existing_inverse=False,
    save_inverse_predictions=True,
)

RUN_EVALUATION = True

if RUN_EVALUATION:
    outputs = run_evaluation_pipeline(CONFIG)
    outputs
else:
    print("RUN_EVALUATION=False")