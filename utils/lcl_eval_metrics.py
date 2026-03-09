from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve


def expected_calibration_error(probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 15) -> float:
    conf, pred = probs.max(dim=1)
    acc = (pred == labels).float()
    ece = 0.0
    bins = torch.linspace(0, 1, int(n_bins) + 1, device=probs.device)
    for i in range(int(n_bins)):
        lo, hi = bins[i], bins[i + 1]
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.float().mean() * (acc[m].mean() - conf[m].mean()).abs()
    return float(ece)


@torch.no_grad()
def evaluate_lcl(
    dl,
    den,
    emb,
    ams,
    device: torch.device,
    sigma_val: float,
    calc_ece: bool = True,
    T_eval: float = 1.0,
) -> Dict[str, Any]:
    den.eval()
    emb.eval()
    if hasattr(den, "train"):
        den.train(False)

    embs: List[torch.Tensor] = []
    labels_dev: List[torch.Tensor] = []
    labels_model: List[torch.Tensor] = []
    logits_inf_all: List[torch.Tensor] = []
    paths_all: List[str] = []

    for x, y_dev, y_model, paths in dl:
        x = x.to(device, non_blocking=True)
        y_dev = y_dev.to(device, non_blocking=True)
        y_model = y_model.to(device, non_blocking=True)
        sigma = torch.full((x.size(0), 1, 1, 1), float(sigma_val), device=device, dtype=x.dtype)
        r = den(x, sigma)
        z = emb(r)
        embs.append(z)
        labels_dev.append(y_dev)
        labels_model.append(y_model)
        paths_all += list(paths)

        logits_inf = ams.inference_logits(z) if hasattr(ams, "inference_logits") else None
        if logits_inf is None:
            # fallback
            W = F.normalize(ams.weight.detach(), dim=1)
            logits_inf = (z @ W.t()) * float(getattr(ams, "s", 1.0))
        logits_inf_all.append(logits_inf)

    embs_t = torch.cat(embs, dim=0)
    labels = torch.cat(labels_dev, dim=0)
    labels_model_t = torch.cat(labels_model, dim=0)
    logits_inf_t = torch.cat(logits_inf_all, dim=0)

    sims = embs_t @ embs_t.t()
    same = labels[:, None] == labels[None, :]
    k_for_nn = min(3, sims.size(1))
    tk = sims.topk(k_for_nn, dim=1)
    top1_idx = tk.indices[:, 1] if k_for_nn >= 2 else tk.indices[:, 0]
    nn_top1 = float((labels[top1_idx] == labels).float().mean().item())
    nn_sim_top1 = tk.values[:, 1] if k_for_nn >= 2 else tk.values[:, 0]
    nn_sim_top2 = tk.values[:, 2] if k_for_nn >= 3 else torch.zeros_like(nn_sim_top1)

    mask_ut = torch.triu(torch.ones_like(sims, dtype=torch.bool), diagonal=1)
    y_true_bin = same[mask_ut].float().cpu().numpy()
    y_score = sims[mask_ut].float().cpu().numpy()
    auc = float(roc_auc_score(y_true_bin, y_score)) if y_true_bin.size > 0 else 0.0
    fpr, tpr, _thr = roc_curve(y_true_bin, y_score) if y_true_bin.size > 0 else (np.array([]), np.array([]), np.array([]))

    logits_train = ams(embs_t, labels)
    ams_ce = float(F.cross_entropy(logits_train, labels).item())

    probs = F.softmax(logits_inf_t / float(T_eval), dim=1)
    pred = probs.argmax(dim=1)
    ams_top1 = float((pred == labels).float().mean().item())
    maxprob = probs.max(dim=1).values
    wrong = pred != labels
    wrong_conf = float(maxprob[wrong].mean().item()) if wrong.any() else 0.0

    dev2model = getattr(dl.dataset, "devcls2modelcls", None)
    y_true_model = labels_model_t
    if dev2model is not None:
        y_pred_model = torch.tensor([dev2model[int(d)] for d in pred.tolist()], device=pred.device)
        y_pred_nn_dev = labels[top1_idx]
        y_pred_nn_model = torch.tensor([dev2model[int(d)] for d in y_pred_nn_dev.tolist()], device=pred.device)
    else:
        y_pred_model = torch.zeros_like(labels_model_t)
        y_pred_nn_model = torch.zeros_like(labels_model_t)

    ece = expected_calibration_error(probs, labels, n_bins=15) if calc_ece else None

    top2_vals = probs.topk(2, dim=1).values
    margin_wrong = float((top2_vals[wrong, 0] - top2_vals[wrong, 1]).mean().item()) if wrong.any() else 0.0

    same_mask_offdiag = same & torch.eye(sims.size(0), device=sims.device).logical_not()
    diff_mask_ut = (~same) & torch.ones_like(sims, dtype=torch.bool, device=sims.device).triu(1)

    return {
        "nn_top1": nn_top1,
        "auc": auc,
        "roc_fpr": fpr,
        "roc_tpr": tpr,
        "same_mean": float(sims[same_mask_offdiag].mean().item()) if same_mask_offdiag.any() else 0.0,
        "diff_mean": float(sims[diff_mask_ut].mean().item()) if diff_mask_ut.any() else 0.0,
        "ams_ce": ams_ce,
        "ams_top1": ams_top1,
        "wrong_conf": wrong_conf,
        "ece": ece,
        "num_wrong": int(wrong.sum().item()),
        "margin_wrong": margin_wrong,
        "N": int(labels.numel()),
        "y_true": labels.detach().cpu().numpy(),
        "y_pred": pred.detach().cpu().numpy(),
        "y_pred_nn": labels[top1_idx].detach().cpu().numpy(),
        "nn_sim_top1": nn_sim_top1.detach().cpu().numpy(),
        "nn_sim_top2": nn_sim_top2.detach().cpu().numpy(),
        "class_names": [dl.dataset.id2dev[i] for i in range(len(dl.dataset.id2dev))],
        "class_names_model": [dl.dataset.id2model[i] for i in range(len(dl.dataset.id2model))] if hasattr(dl.dataset, "id2model") else [],
        "embs": embs_t.detach().cpu().numpy(),
        "paths": paths_all,
        "prob_max": probs.max(dim=1).values.detach().cpu().numpy(),
        "prob_second": probs.topk(2, dim=1).values[:, 1].detach().cpu().numpy(),
        "y_true_model": y_true_model.detach().cpu().numpy(),
        "y_pred_model": y_pred_model.detach().cpu().numpy(),
        "y_pred_nn_model": y_pred_nn_model.detach().cpu().numpy(),
    }

