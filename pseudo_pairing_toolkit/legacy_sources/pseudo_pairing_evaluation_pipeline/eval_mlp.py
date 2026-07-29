"""Forward and inverse MLP evaluation pipeline for repeated pseudo-control datasets."""
from __future__ import annotations

import gc
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    hamming_loss,
    jaccard_score,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
    top_k_accuracy_score,
)
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from eval_common import (
    add_metadata_columns,
    as_namespace,
    ensure_dir,
    get_config,
    load_pseudo_matrix_aligned,
    load_run_manifest,
    make_safe_id,
    mean_axis0,
    metadata_from_manifest_row,
    rowwise_cosine,
    rowwise_pearson,
    safe_cosine,
    safe_pearson,
    safe_spearman,
    save_json,
    select_eval_genes,
    set_seed,
    slice_to_dense,
    summarize_array,
    summarize_numeric,
    to_csr,
)
from mlp_models import build_forward_model, build_inverse_model, InverseMultiLabelGeneMLP


def device_from_config(config) -> torch.device:
    requested = get_config(config, "device", "auto")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(str(requested))


class IndexDataset(Dataset):
    def __init__(self, indices):
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        return int(self.indices[i])


def build_label_encoder(perturbed, perturbation_key: str, min_cells: int, outdir: Path):
    labels_raw = perturbed.obs[perturbation_key].astype(str).values
    counts = pd.Series(labels_raw).value_counts()
    eligible = sorted(counts[counts >= int(min_cells)].index.astype(str).tolist())
    label_to_id = {lab: i for i, lab in enumerate(eligible)}
    id_to_label = {i: lab for lab, i in label_to_id.items()}
    y = np.full(perturbed.n_obs, -1, dtype=np.int64)
    for lab, i in label_to_id.items():
        y[labels_raw == lab] = i
    enc = pd.DataFrame({"perturbation_label": eligible, "perturbation_id": [label_to_id[x] for x in eligible], "n_cells": [int(counts.loc[x]) for x in eligible]})
    enc.to_csv(outdir / "perturbation_label_encoder.csv", index=False)
    return labels_raw, y, label_to_id, id_to_label


def build_fixed_split(labels_raw: np.ndarray, y_all: np.ndarray, config, split_dir: Path):
    split_path = split_dir / "fixed_cell_index_split.npz"
    if split_path.exists() and bool(get_config(config, "reuse_existing_split", True)):
        data = np.load(split_path)
        return data["train_idx"], data["val_idx"], data["test_idx"]
    rng = np.random.default_rng(int(get_config(config, "split_seed", 42)))
    train_frac = float(get_config(config, "train_frac", 0.70))
    val_frac = float(get_config(config, "val_frac", 0.15))
    train_parts, val_parts, test_parts = [], [], []
    for lab in sorted(pd.Index(labels_raw[y_all >= 0]).unique().astype(str).tolist()):
        idx = np.where((labels_raw == lab) & (y_all >= 0))[0]
        if len(idx) < 3:
            continue
        rng.shuffle(idx)
        n = len(idx)
        n_train = max(1, int(np.floor(n * train_frac)))
        n_val = max(1, int(np.floor(n * val_frac)))
        if n - n_train - n_val <= 0:
            n_val = max(1, min(n - 2, n_val))
            n_train = max(1, n - n_val - 1)
        train_parts.append(idx[:n_train])
        val_parts.append(idx[n_train:n_train + n_val])
        test_parts.append(idx[n_train + n_val:])
    train_idx = np.concatenate(train_parts).astype(np.int64)
    val_idx = np.concatenate(val_parts).astype(np.int64)
    test_idx = np.concatenate(test_parts).astype(np.int64)
    rng.shuffle(train_idx); rng.shuffle(val_idx); rng.shuffle(test_idx)
    np.savez(split_path, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)
    pd.DataFrame({"split": ["train", "val", "test"], "n_cells": [len(train_idx), len(val_idx), len(test_idx)]}).to_csv(split_dir / "fixed_split_summary.csv", index=False)
    return train_idx, val_idx, test_idx


def make_pair_collate_fn(X0, Xt, pert_ids, task: str, control_mean_vec=None, inverse_input_mode: str = "strategy_delta"):
    if control_mean_vec is not None:
        control_mean_vec = np.asarray(control_mean_vec, dtype=np.float32).reshape(1, -1)

    def collate_fn(batch_indices):
        idx = np.asarray(batch_indices, dtype=np.int64)
        xt = slice_to_dense(Xt, idx)
        t = pert_ids[idx].astype(np.int64)
        if task == "forward":
            x0 = slice_to_dense(X0, idx)
            return torch.from_numpy(x0), torch.from_numpy(t), torch.from_numpy(xt - x0), torch.from_numpy(xt)
        if task == "inverse":
            if inverse_input_mode == "strategy_delta":
                x0 = slice_to_dense(X0, idx)
                delta = xt - x0
            elif inverse_input_mode == "common_delta":
                if control_mean_vec is None:
                    raise ValueError("control_mean_vec is required for common_delta")
                delta = xt - control_mean_vec
            else:
                raise ValueError(f"Unknown inverse_input_mode={inverse_input_mode}")
            return torch.from_numpy(np.asarray(delta, dtype=np.float32)), torch.from_numpy(t)
        raise ValueError(f"Unknown task={task}")
    return collate_fn


def make_loaders(indices, X0, Xt, pert_ids, config, task: str, control_mean_vec=None, inverse_input_mode="strategy_delta"):
    batch_size = int(get_config(config, "batch_size", 256))
    num_workers = int(get_config(config, "num_workers", 0))
    train_idx, val_idx, test_idx = indices
    return (
        DataLoader(IndexDataset(train_idx), batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=False,
                   collate_fn=make_pair_collate_fn(X0, Xt, pert_ids, task, control_mean_vec, inverse_input_mode)),
        DataLoader(IndexDataset(val_idx), batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False,
                   collate_fn=make_pair_collate_fn(X0, Xt, pert_ids, task, control_mean_vec, inverse_input_mode)),
        DataLoader(IndexDataset(test_idx), batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False,
                   collate_fn=make_pair_collate_fn(X0, Xt, pert_ids, task, control_mean_vec, inverse_input_mode)),
    )


def train_forward_model(model, train_loader, val_loader, run_id: str, outdir: Path, config, device: torch.device):
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(get_config(config, "learning_rate", 1e-3)), weight_decay=float(get_config(config, "weight_decay", 1e-5)))
    best_val = np.inf; wait = 0; hist = []
    best_path = outdir / f"{run_id}_forward_best.pt"
    for epoch in tqdm(range(1, int(get_config(config, "forward_epochs", 30)) + 1), desc=f"{run_id} forward"):
        model.train(); tr = []
        for x0, t, delta_true, _xt in train_loader:
            x0 = x0.to(device); t = t.to(device); delta_true = delta_true.to(device)
            opt.zero_grad(); delta_pred = model(x0, t); loss = F.mse_loss(delta_pred, delta_true)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), float(get_config(config, "max_grad_norm", 5.0))); opt.step()
            tr.append(loss.item())
        model.eval(); va = []
        with torch.no_grad():
            for x0, t, delta_true, _xt in val_loader:
                x0 = x0.to(device); t = t.to(device); delta_true = delta_true.to(device)
                va.append(F.mse_loss(model(x0, t), delta_true).item())
        row = {"epoch": epoch, "train_loss": float(np.mean(tr)), "val_loss": float(np.mean(va))}
        hist.append(row)
        if row["val_loss"] < best_val:
            best_val = row["val_loss"]; wait = 0; torch.save(model.state_dict(), best_path)
        else:
            wait += 1
        if wait >= int(get_config(config, "early_stop_patience", 3)):
            break
    hist_df = pd.DataFrame(hist)
    hist_df.to_csv(outdir / f"{run_id}_forward_training_history.csv", index=False)
    model.load_state_dict(torch.load(best_path, map_location=device))
    return model, hist_df


def train_inverse_model(model, train_loader, val_loader, run_id: str, outdir: Path, config, device: torch.device):
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(get_config(config, "learning_rate", 1e-3)), weight_decay=float(get_config(config, "weight_decay", 1e-5)))
    best_val = np.inf; wait = 0; hist = []
    best_path = outdir / f"{run_id}_inverse_best.pt"
    for epoch in tqdm(range(1, int(get_config(config, "inverse_epochs", 30)) + 1), desc=f"{run_id} inverse"):
        model.train(); tr = []
        for delta, t in train_loader:
            delta = delta.to(device); t = t.to(device)
            opt.zero_grad(); logits = model(delta); loss = F.cross_entropy(logits, t)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), float(get_config(config, "max_grad_norm", 5.0))); opt.step()
            tr.append(loss.item())
        model.eval(); va = []
        with torch.no_grad():
            for delta, t in val_loader:
                delta = delta.to(device); t = t.to(device)
                va.append(F.cross_entropy(model(delta), t).item())
        row = {"epoch": epoch, "train_loss": float(np.mean(tr)), "val_loss": float(np.mean(va))}
        hist.append(row)
        if row["val_loss"] < best_val:
            best_val = row["val_loss"]; wait = 0; torch.save(model.state_dict(), best_path)
        else:
            wait += 1
        if wait >= int(get_config(config, "early_stop_patience", 3)):
            break
    hist_df = pd.DataFrame(hist)
    hist_df.to_csv(outdir / f"{run_id}_inverse_training_history.csv", index=False)
    model.load_state_dict(torch.load(best_path, map_location=device))
    return model, hist_df


def finalize_per_perturbation_common_delta(sum_pred, sum_true, counts, id_to_label, topk_list=(20, 50, 100)) -> pd.DataFrame:
    recs = []
    for cls in range(len(counts)):
        n = int(counts[cls])
        if n <= 0:
            continue
        pred = sum_pred[cls] / n; true = sum_true[cls] / n
        rec = {
            "perturbation_id": cls,
            "perturbation_label": id_to_label.get(cls, str(cls)),
            "n_test_cells": n,
            "common_delta_pearson": safe_pearson(pred, true),
            "common_delta_spearman": safe_spearman(pred, true),
            "common_delta_cosine": safe_cosine(pred, true),
            "common_delta_l2": float(np.sqrt(np.mean((pred - true) ** 2))),
        }
        for k in topk_list:
            kk = min(int(k), len(true))
            true_idx = np.argsort(np.abs(true))[-kk:]
            pred_idx = np.argsort(np.abs(pred))[-kk:]
            rec[f"top{kk}_common_delta_pearson_true_genes"] = safe_pearson(pred[true_idx], true[true_idx])
            rec[f"top{kk}_common_delta_cosine_true_genes"] = safe_cosine(pred[true_idx], true[true_idx])
            rec[f"top{kk}_common_delta_overlap_fraction"] = float(len(set(true_idx.tolist()).intersection(set(pred_idx.tolist()))) / max(kk, 1))
        recs.append(rec)
    return pd.DataFrame(recs)


def evaluate_forward_model(model, test_loader, control_mean, n_classes, id_to_label, config, device):
    model.eval()
    control_mean = np.asarray(control_mean, dtype=np.float32).reshape(1, -1)
    sums = {"model_sse_xt": 0.0, "model_sae_xt": 0.0, "input_sse_xt": 0.0, "input_sae_xt": 0.0, "delta_sse": 0.0, "delta_sae": 0.0}
    n_elem = 0
    corr = {"model_cell_pearson_xt": [], "input_cell_pearson_xt": [], "model_cell_cosine_xt": [], "input_cell_cosine_xt": [], "strategy_delta_cell_pearson": [], "strategy_delta_cell_cosine": [], "model_common_delta_cell_pearson": [], "input_common_delta_cell_pearson": []}
    sum_pred = np.zeros((n_classes, control_mean.shape[1]), dtype=np.float64)
    sum_true = np.zeros((n_classes, control_mean.shape[1]), dtype=np.float64)
    counts = np.zeros(n_classes, dtype=np.int64)
    with torch.no_grad():
        for x0, t, delta_true, xt_true in tqdm(test_loader, desc="Evaluate forward", leave=False):
            t_np = t.numpy().astype(np.int64)
            x0_gpu = x0.to(device); t_gpu = t.to(device)
            delta_pred = model(x0_gpu, t_gpu).detach().cpu().numpy().astype(np.float32)
            x0_np = x0.numpy().astype(np.float32)
            delta_true_np = delta_true.numpy().astype(np.float32)
            xt_true_np = xt_true.numpy().astype(np.float32)
            xt_pred = x0_np + delta_pred
            model_err = xt_pred - xt_true_np; input_err = x0_np - xt_true_np; delta_err = delta_pred - delta_true_np
            sums["model_sse_xt"] += float(np.sum(model_err ** 2)); sums["model_sae_xt"] += float(np.sum(np.abs(model_err)))
            sums["input_sse_xt"] += float(np.sum(input_err ** 2)); sums["input_sae_xt"] += float(np.sum(np.abs(input_err)))
            sums["delta_sse"] += float(np.sum(delta_err ** 2)); sums["delta_sae"] += float(np.sum(np.abs(delta_err)))
            n_elem += int(np.prod(xt_true_np.shape))
            corr["model_cell_pearson_xt"].extend(rowwise_pearson(xt_pred, xt_true_np).tolist())
            corr["input_cell_pearson_xt"].extend(rowwise_pearson(x0_np, xt_true_np).tolist())
            corr["model_cell_cosine_xt"].extend(rowwise_cosine(xt_pred, xt_true_np).tolist())
            corr["input_cell_cosine_xt"].extend(rowwise_cosine(x0_np, xt_true_np).tolist())
            corr["strategy_delta_cell_pearson"].extend(rowwise_pearson(delta_pred, delta_true_np).tolist())
            corr["strategy_delta_cell_cosine"].extend(rowwise_cosine(delta_pred, delta_true_np).tolist())
            true_common = xt_true_np - control_mean; pred_common = xt_pred - control_mean; input_common = x0_np - control_mean
            corr["model_common_delta_cell_pearson"].extend(rowwise_pearson(pred_common, true_common).tolist())
            corr["input_common_delta_cell_pearson"].extend(rowwise_pearson(input_common, true_common).tolist())
            for cls in np.unique(t_np):
                if cls < 0: continue
                mask = t_np == cls
                sum_pred[cls] += pred_common[mask].sum(axis=0); sum_true[cls] += true_common[mask].sum(axis=0); counts[cls] += int(mask.sum())
    m = {
        "model_mse_xt": sums["model_sse_xt"] / max(n_elem, 1),
        "model_mae_xt": sums["model_sae_xt"] / max(n_elem, 1),
        "input_only_mse_xt": sums["input_sse_xt"] / max(n_elem, 1),
        "input_only_mae_xt": sums["input_sae_xt"] / max(n_elem, 1),
        "strategy_specific_delta_mse": sums["delta_sse"] / max(n_elem, 1),
        "strategy_specific_delta_mae": sums["delta_sae"] / max(n_elem, 1),
    }
    m["model_gain_mse_xt"] = m["input_only_mse_xt"] - m["model_mse_xt"]
    m["model_gain_mae_xt"] = m["input_only_mae_xt"] - m["model_mae_xt"]
    m["model_gain_mse_xt_fraction"] = m["model_gain_mse_xt"] / (m["input_only_mse_xt"] + 1e-12)
    for k, vals in corr.items():
        m.update(summarize_array(vals, k))
    per = finalize_per_perturbation_common_delta(sum_pred, sum_true, counts, id_to_label, topk_list=get_config(config, "topk_effect_genes", [20, 50, 100]))
    for col in [c for c in per.columns if c not in {"perturbation_id", "perturbation_label", "n_test_cells"} and pd.api.types.is_numeric_dtype(per[c])]:
        arr = per[col].astype(float).values
        m[f"per_perturbation_{col}_mean"] = float(np.nanmean(arr)) if arr.size else np.nan
        m[f"per_perturbation_{col}_median"] = float(np.nanmedian(arr)) if arr.size else np.nan
    m["n_perturbations_evaluated"] = int(per.shape[0])
    return m, per



def safe_multiclass_ovr_auc(
    y_true: np.ndarray,
    prob: np.ndarray,
    labels: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray]:
    """Compute robust one-vs-rest macro/weighted AUC for multiclass prediction.

    sklearn's multiclass roc_auc_score can fail when the test split does not contain
    all classes. This helper computes per-class one-vs-rest AUCs and averages only
    classes with both positive and negative test examples.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    prob = np.asarray(prob, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)

    class_auc = np.full(labels.shape[0], np.nan, dtype=np.float64)
    support = np.zeros(labels.shape[0], dtype=np.int64)

    for j, lab in enumerate(labels):
        y_binary = (y_true == int(lab)).astype(np.int64)
        support[j] = int(y_binary.sum())

        # AUC is undefined if the class is absent or if all test cells belong to it.
        if y_binary.sum() == 0 or y_binary.sum() == y_binary.shape[0]:
            continue

        try:
            class_auc[j] = float(roc_auc_score(y_binary, prob[:, j]))
        except ValueError:
            class_auc[j] = np.nan

    valid = np.isfinite(class_auc)
    metrics: dict[str, Any] = {
        "test_macro_auc_ovr": float(np.nanmean(class_auc)) if valid.any() else np.nan,
        "test_weighted_auc_ovr": float(np.average(class_auc[valid], weights=support[valid])) if valid.any() and support[valid].sum() > 0 else np.nan,
        "test_macro_auc_ovr_n_valid_classes": int(valid.sum()),
        "test_macro_auc_ovr_n_total_classes": int(labels.shape[0]),
    }
    return metrics, class_auc


def evaluate_inverse_model(model, test_loader, n_classes, id_to_label, config, device):
    model.eval(); ys = []; ps = []; probs = []
    with torch.no_grad():
        for delta, t in tqdm(test_loader, desc="Evaluate inverse", leave=False):
            prob = torch.softmax(model(delta.to(device)), dim=1).detach().cpu().numpy()
            ys.append(t.numpy().astype(np.int64)); ps.append(np.argmax(prob, axis=1).astype(np.int64)); probs.append(prob.astype(np.float32))
    y_true = np.concatenate(ys); y_pred = np.concatenate(ps); prob = np.vstack(probs)
    labels = np.arange(n_classes)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    auc_metrics, per_class_auc = safe_multiclass_ovr_auc(y_true, prob, labels)
    metrics = {
        "inverse_task_type": "combination_multiclass",
        "test_n_cells": int(len(y_true)),
        "test_n_classes": int(n_classes),
        "test_cross_entropy": float(log_loss(y_true, prob, labels=labels)),
        "test_accuracy": float(accuracy_score(y_true, y_pred)),
        "test_balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "test_macro_precision": float(macro_p), "test_macro_recall": float(macro_r), "test_macro_f1": float(macro_f1),
        "test_weighted_precision": float(weighted_p), "test_weighted_recall": float(weighted_r), "test_weighted_f1": float(weighted_f1),
    }
    metrics.update(auc_metrics)
    for k in get_config(config, "inverse_topk_accuracy_list", [3, 5, 10]):
        if n_classes >= int(k):
            metrics[f"test_top{int(k)}_accuracy"] = float(top_k_accuracy_score(y_true, prob, k=int(k), labels=labels))
    pc_p, pc_r, pc_f1, pc_sup = precision_recall_fscore_support(y_true, y_pred, labels=labels, average=None, zero_division=0)
    per = pd.DataFrame({
        "perturbation_id": labels.astype(int),
        "perturbation_label": [id_to_label.get(int(i), str(i)) for i in labels],
        "test_precision": pc_p,
        "test_recall": pc_r,
        "test_f1": pc_f1,
        "test_support": pc_sup,
        "test_auc_ovr": per_class_auc,
    })
    pred = {"y_true": y_true, "y_pred": y_pred}
    if bool(get_config(config, "return_inverse_probabilities", False)):
        pred["prob"] = prob
    return metrics, per, pred


def train_and_evaluate_forward(run_id, X0, Xt, y_all, indices, control_mean, n_classes, id_to_label, metadata, config, outdir, device):
    loaders = make_loaders(indices, X0, Xt, y_all, config, task="forward")
    set_seed(int(get_config(config, "model_seed", 42)))
    model = build_forward_model(Xt.shape[1], n_classes, config)
    model, _ = train_forward_model(model, loaders[0], loaders[1], run_id, outdir, config, device)
    metrics, per = evaluate_forward_model(model, loaders[2], control_mean, n_classes, id_to_label, config, device)
    metrics.update(metadata)
    per = add_metadata_columns(per, metadata)
    save_json(metrics, outdir / f"{run_id}_forward_test_metrics.json")
    per.to_csv(outdir / f"{run_id}_forward_per_perturbation_common_delta.csv", index=False)
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return metrics, per


def train_and_evaluate_inverse(run_id, X0, Xt, y_all, indices, control_mean, n_classes, id_to_label, metadata, config, outdir, device, inverse_input_mode="strategy_delta"):
    loaders = make_loaders(indices, X0, Xt, y_all, config, task="inverse", control_mean_vec=control_mean, inverse_input_mode=inverse_input_mode)
    set_seed(int(get_config(config, "model_seed", 42)))
    model = build_inverse_model(Xt.shape[1], n_classes, config)
    model, _ = train_inverse_model(model, loaders[0], loaders[1], run_id, outdir, config, device)
    metrics, per, pred = evaluate_inverse_model(model, loaders[2], n_classes, id_to_label, config, device)
    metrics.update(metadata); metrics["inverse_input_mode"] = inverse_input_mode
    per = add_metadata_columns(per, metadata); per["inverse_input_mode"] = inverse_input_mode
    save_json(metrics, outdir / f"{run_id}_inverse_classification_metrics.json")
    per.to_csv(outdir / f"{run_id}_inverse_classification_per_class.csv", index=False)
    if bool(get_config(config, "save_inverse_predictions", True)):
        np.savez_compressed(outdir / f"{run_id}_inverse_classification_test_predictions.npz", **pred)
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return metrics, per


def run_mlp_evaluation(config: Mapping[str, Any]) -> dict[str, Path]:
    config = as_namespace(config)
    outdir = ensure_dir(Path(get_config(config, "outdir")) / "downstream_mlp")
    forward_dir = ensure_dir(outdir / "forward_outputs")
    inverse_dir = ensure_dir(outdir / "inverse_classification_outputs")
    split_dir = ensure_dir(Path(get_config(config, "split_dir", outdir / "fixed_splits")))
    runs_df = load_run_manifest(config)
    runs_df.to_csv(outdir / "discovered_run_manifest.csv", index=False)
    device = device_from_config(config)

    control = sc.read_h5ad(str(get_config(config, "control_h5ad"))); control.obs_names_make_unique()
    perturbed = sc.read_h5ad(str(get_config(config, "perturbed_h5ad"))); perturbed.obs_names_make_unique()
    key = get_config(config, "perturbation_key", "perturbation_key")
    if key not in perturbed.obs.columns:
        raise KeyError(f"perturbation_key={key!r} not in perturbed.obs")
    eval_genes = select_eval_genes(control, perturbed, runs_df, max_eval_genes=get_config(config, "max_eval_genes", 3000), check_shared_genes_across_all_runs=False)
    pd.Series(eval_genes).to_csv(outdir / "eval_genes.csv", index=False, header=["gene"])
    labels_raw, y_all, label_to_id, id_to_label = build_label_encoder(perturbed, key, int(get_config(config, "min_cells_per_perturbation", 20)), outdir)
    Xt = to_csr(perturbed[:, eval_genes].X)
    X_control = to_csr(control[:, eval_genes].X)
    control_mean = mean_axis0(X_control).astype(np.float32)
    np.save(outdir / "control_mean_eval_genes.npy", control_mean)
    indices = build_fixed_split(labels_raw, y_all, config, split_dir)

    forward_records, forward_per_records, inverse_records, inverse_per_records = [], [], [], []
    tasks = set(get_config(config, "mlp_tasks", ["forward", "inverse_strategy_delta"]))
    skip_f = bool(get_config(config, "skip_existing_forward", True))
    skip_i = bool(get_config(config, "skip_existing_inverse", True))

    for i, row in tqdm(runs_df.iterrows(), total=runs_df.shape[0], desc="MLP evaluation"):
        metadata = metadata_from_manifest_row(row)
        run_id = str(metadata["run_id"])
        X0 = load_pseudo_matrix_aligned(row, perturbed, eval_genes, require_full_alignment=True)

        if "forward" in tasks:
            jf = forward_dir / f"{run_id}_forward_test_metrics.json"
            pf = forward_dir / f"{run_id}_forward_per_perturbation_common_delta.csv"
            if skip_f and jf.exists() and pf.exists():
                fm = pd.read_json(jf, typ="series").to_dict(); fp = pd.read_csv(pf)
            else:
                fm, fp = train_and_evaluate_forward(run_id, X0, Xt, y_all, indices, control_mean, len(label_to_id), id_to_label, metadata, config, forward_dir, device)
            forward_records.append(fm); forward_per_records.append(fp)
            pd.DataFrame(forward_records).to_csv(outdir / "forward_mlp_run_summary_PARTIAL.csv", index=False)

        if "inverse_strategy_delta" in tasks:
            inv_id = make_safe_id(f"{run_id}__inverse_strategy_delta")
            ji = inverse_dir / f"{inv_id}_inverse_classification_metrics.json"
            pi = inverse_dir / f"{inv_id}_inverse_classification_per_class.csv"
            if skip_i and ji.exists() and pi.exists():
                im = pd.read_json(ji, typ="series").to_dict(); ip = pd.read_csv(pi)
            else:
                im, ip = train_and_evaluate_inverse(inv_id, X0, Xt, y_all, indices, control_mean, len(label_to_id), id_to_label, metadata, config, inverse_dir, device, inverse_input_mode="strategy_delta")
            inverse_records.append(im); inverse_per_records.append(ip)
            pd.DataFrame(inverse_records).to_csv(outdir / "inverse_mlp_run_summary_PARTIAL.csv", index=False)

        del X0; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    if "inverse_common_delta" in tasks:
        metadata = {"run_id": "COMMON_REFERENCE_DELTA", "strategy": "COMMON_REFERENCE_DELTA", "strategy_id": "COMMON_REFERENCE_DELTA", "strategy_family": "common_reference_delta", "parameter_label": "xt_minus_mean_control"}
        inv_id = "COMMON_REFERENCE_DELTA__inverse_common_delta"
        im, ip = train_and_evaluate_inverse(inv_id, Xt, Xt, y_all, indices, control_mean, len(label_to_id), id_to_label, metadata, config, inverse_dir, device, inverse_input_mode="common_delta")
        inverse_records.append(im); inverse_per_records.append(ip)

    paths = {}
    if forward_records:
        fdf = pd.DataFrame(forward_records); fpath = outdir / "forward_mlp_run_summary.csv"; fdf.to_csv(fpath, index=False); paths["forward_summary"] = fpath
        if forward_per_records:
            fppath = outdir / "forward_mlp_per_perturbation_common_delta.csv"; pd.concat(forward_per_records, ignore_index=True).to_csv(fppath, index=False); paths["forward_per_perturbation"] = fppath
        summarize_numeric(fdf, ["perturbed_group", "strategy", "strategy_family"], outdir / "summaries/forward_seed_averaged_by_strategy.csv")
    if inverse_records:
        idf = pd.DataFrame(inverse_records); ipath = outdir / "inverse_mlp_run_summary.csv"; idf.to_csv(ipath, index=False); paths["inverse_summary"] = ipath
        if inverse_per_records:
            icpath = outdir / "inverse_mlp_per_class.csv"; pd.concat(inverse_per_records, ignore_index=True).to_csv(icpath, index=False); paths["inverse_per_class"] = icpath
        group_cols = [c for c in ["perturbed_group", "strategy", "strategy_family", "inverse_input_mode"] if c in idf.columns]
        summarize_numeric(idf, group_cols, outdir / "summaries/inverse_seed_averaged_by_strategy.csv")
    save_json({"n_runs": len(runs_df), "n_eval_genes": len(eval_genes), "n_perturbation_classes": len(label_to_id), "device": str(device)}, outdir / "mlp_config_summary.json")
    return paths
