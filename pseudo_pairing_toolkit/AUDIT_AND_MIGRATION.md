# Refactor audit and migration map

## Original-to-package mapping

| Uploaded component | Packaged module |
|---|---|
| `pseudo_pairing_refactor/pseudo_pairing_utils.py` | `pseudopair.pairing.utils` |
| `pseudo_pairing_refactor/pseudo_pairing_seacell.py` | `pseudopair.pairing.seacell` |
| `pseudo_pairing_refactor/pseudo_pairing_strategies.py` | `pseudopair.pairing.strategies` |
| `pseudo_pairing_refactor/run_pseudo_pairing_repetitions.py` | `pseudopair.pairing.runner` |
| `pseudo_pairing_evaluation_pipeline/eval_common.py` | `pseudopair.evaluation.common` |
| `eval_control_manifold.py` | `pseudopair.evaluation.control_manifold` |
| `eval_perturbation_effect.py` | `pseudopair.evaluation.perturbation_effect` |
| `eval_mlp.py` | `pseudopair.evaluation.mlp` |
| `mlp_models.py` | `pseudopair.evaluation.models` |
| result-analysis scripts | `pseudopair.analysis.*` |
| `data_preprocessing.ipynb` | `pseudopair.preprocessing` |

## Structural changes

- Replaced hard-coded `WORKFLOW_DIR` and `sys.path.insert(...)` execution with package-relative imports.
- Added YAML/JSON configuration loading, environment-variable expansion, and deterministic path resolution.
- Added one CLI for all stages.
- Added generic URL/local-file acquisition with optional SHA-256 verification.
- Converted the active Replogle preprocessing logic into configurable functions.
- Added automatic path handoff from preprocessing to pairing and evaluation.
- Added synchronous stage-state recording and resolved-configuration snapshots.
- Preserved the existing per-run manifests, QC files, skip-existing behavior, and seed-only aggregation logic.
- Removed one obsolete duplicate definition of `detect_existing_pseudo_control_output`; the later, strategy-aware implementation was already the effective runtime definition.

## Scientific logic intentionally preserved

No formulas or evaluation definitions were redesigned. The transferred implementation retains:

- S0–S5 strategy construction;
- SEACell membership reuse;
- entropically regularized OT and top-k metacell handling;
- true-control sampling within selected metacells for S5;
- control-manifold metrics;
- perturbation-effect metrics;
- forward and inverse MLP training/evaluation;
- canonical strategy-variant parsing;
- averaging over random seeds only;
- S0/S1 and inverse-classification reference lines.

## Verification completed here

- All Python files pass byte-code compilation.
- The CLI validates both included example configurations.
- Relative and nested h5ad path resolution is unit-tested.
- The package import graph was tested with lightweight Scanpy/AnnData stubs.
- Two configuration tests pass.
- Editable installation succeeds with build isolation disabled in the restricted packaging environment.

## Verification still required on the scientific environment

The packaging environment used for this refactor does not contain Scanpy, AnnData, SEACells, POT, or the actual h5ad datasets. Therefore, full numerical regression against existing outputs could not be executed here. Run `examples/smoke_test_scientific.py`, then compare one real-dataset S0–S5 subset against the existing manifest and metric tables before replacing the production workflow.

## Foundation-model encoder branch

The uploaded `Tokenization_FMs` branch is distributed as the sibling top-level folder `../foundation_model_encoders/`. The encoder mathematics and downstream MLP task definitions were retained. Machine-specific repository, checkpoint, data, output, and Python-environment paths were moved into `configs/encoders.example.yaml` and resolved by `launcher.py`.

The maintained scripts now obtain their paths through `PPFM_*` environment variables set by the launcher. The legacy notebooks are retained for provenance. The MLP resolver supports configurable output roots and includes Geneformer as an embedding representation. UMAP discovery now loads the metadata filenames emitted by all encoder scripts and iterates over every selected model rather than a single hard-coded representation.
