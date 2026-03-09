from __future__ import annotations

import argparse
import math
import os
import random
import warnings
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.LCLNet import LCLNet
from utils.lcl_ckpt import find_latest_ckpt, load_weights
from utils.lcl_dataset import QuadDataset
from utils.lcl_losses import HardTripletLoss, HierQuadLoss


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_seed_worker(base_seed: int):
    def _seed_worker(worker_id: int):
        seed = (base_seed + worker_id) % (2**32)
        np.random.seed(seed)
        random.seed(seed)

    return _seed_worker


@torch.no_grad()
def evaluate(
    dl: DataLoader,
    net: LCLNet,
    device: torch.device,
    sigma: float,
) -> Dict[str, float]:
    net.eval()

    embs = []
    labels = []
    for batch in dl:
        a = batch[0].to(device, non_blocking=True)
        y = batch[-1].to(device, non_blocking=True)
        net.sigma_value = float(sigma)
        ea, _, _, _, _ = net(a, label=None, denoise_no_grad=True)
        embs.append(ea)
        labels.append(y)

    embs_t = torch.cat(embs, dim=0)
    labels_t = torch.cat(labels, dim=0)

    sims = F.linear(embs_t, embs_t)
    same = labels_t[:, None] == labels_t[None, :]

    top1_idx = sims.topk(2, 1).indices[:, 1]
    nn_top1 = (labels_t[top1_idx] == labels_t).float().mean().item()

    mask_ut = torch.triu(torch.ones_like(sims, dtype=torch.bool), 1)
    y_true = same[mask_ut].cpu().numpy().astype(int)
    y_score = sims[mask_ut].detach().cpu().numpy()
    auc = float(roc_auc_score(y_true, y_score)) if y_true.size > 0 else 0.0

    same_ut = same & mask_ut
    diff_ut = (~same) & mask_ut
    same_mean = float(sims[same_ut].mean().item()) if same_ut.any() else 0.0
    diff_mean = float(sims[diff_ut].mean().item()) if diff_ut.any() else 0.0

    logits_train = net.ams(embs_t, labels_t)
    ams_ce = float(F.cross_entropy(logits_train, labels_t).item())
    pred_train = logits_train.argmax(dim=1)
    ams_top1_train = float((pred_train == labels_t).float().mean().item())

    # Inference-style logits (no margin) for diagnostics.
    logits_inf = net.ams.inference_logits(embs_t)
    pred_inf = logits_inf.argmax(dim=1)
    ams_top1 = float((pred_inf == labels_t).float().mean().item())
    probs = F.softmax(logits_inf, dim=1)
    wrong = pred_inf != labels_t
    wrong_conf = float(probs.max(dim=1).values[wrong].mean().item()) if wrong.any() else 0.0

    var = float(embs_t.var(dim=0).mean().item())

    return {
        "nn_top1": nn_top1,
        "auc": auc,
        "same_mean": same_mean,
        "diff_mean": diff_mean,
        "var": var,
        "ams_ce": ams_ce,
        "ams_top1_train": ams_top1_train,
        "ams_top1": ams_top1,
        "wrong_conf": wrong_conf,
    }


def train(opt: argparse.Namespace) -> None:
    seed_everything(opt.seed)
    device = torch.device(opt.device if (opt.device.startswith("cuda") and torch.cuda.is_available()) else "cpu")
    print(f"[Device] {device}")

    ds_tr = QuadDataset(opt.csv, opt.root, opt.crop, opt.resize)
    dl_tr = DataLoader(
        ds_tr,
        batch_size=opt.bs,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
        persistent_workers=(opt.num_workers > 0),
        prefetch_factor=opt.prefetch_factor if opt.num_workers > 0 else None,
        worker_init_fn=make_seed_worker(opt.seed),
        generator=torch.Generator().manual_seed(opt.seed),
    )

    dl_va: Optional[DataLoader] = None
    if opt.csv_val:
        ds_va = QuadDataset(opt.csv_val, opt.root, opt.crop, opt.resize)
        dl_va = DataLoader(
            ds_va,
            batch_size=opt.bs,
            shuffle=False,
            num_workers=opt.num_workers,
            pin_memory=True,
            persistent_workers=(opt.num_workers > 0),
            prefetch_factor=opt.prefetch_factor if opt.num_workers > 0 else None,
            worker_init_fn=make_seed_worker(opt.seed + 1),
            generator=torch.Generator().manual_seed(opt.seed + 1),
        )

    net = LCLNet(
        num_classes=ds_tr.cls_num,
        out_dim=opt.out_dim,
        sigma=opt.sigma,
        ams_s=opt.ams_s,
        ams_m=opt.ams_m,
    ).to(device)
    print(f"[Classes] {ds_tr.cls_num}")

    quad = HierQuadLoss(opt.m1, opt.m2, opt.lambda_m).to(device)
    trip = HardTripletLoss(opt.trip_margin).to(device)

    # Optional: resume
    ckpt_path = None
    if opt.pretrained:
        if opt.pretrained.lower() == "auto":
            ckpt_path = find_latest_ckpt(opt.save_dir)
        else:
            ckpt_path = opt.pretrained if os.path.isfile(opt.pretrained) else find_latest_ckpt(opt.pretrained)

    if ckpt_path and os.path.isfile(ckpt_path):
        print(f"[LOAD] {ckpt_path}")
        state = torch.load(ckpt_path, map_location="cpu")
        if isinstance(state, dict):
            if "den" in state:
                net.den.load_state_dict(state["den"], strict=False)
            if "emb" in state:
                net.emb.load_state_dict(state["emb"], strict=False)
            if "ams" in state:
                net.ams.load_state_dict(state["ams"], strict=False)
    else:
        # Try official FFDNet weights (optional)
        if opt.ffdnet_ckpt and os.path.isfile(opt.ffdnet_ckpt):
            info = load_weights(net.den, opt.ffdnet_ckpt, strict=False)
            if info.missing_keys or info.unexpected_keys:
                print(f"[FFDNet] missing={list(info.missing_keys)} unexpected={list(info.unexpected_keys)}")
            else:
                print("[FFDNet] loaded official gray weights.")

    # Freeze denoiser if requested
    if opt.freeze_den:
        for p in net.den.parameters():
            p.requires_grad_(False)
        net.den.eval()

    optm = torch.optim.Adam(
        [
            {"params": net.den.parameters(), "lr": opt.lr_den, "weight_decay": opt.weight_decay},
            {"params": net.emb.parameters(), "lr": opt.lr_emb, "weight_decay": opt.weight_decay},
            {"params": net.ams.parameters(), "lr": opt.lr_ams},
        ]
    )

    scaler = GradScaler(enabled=(opt.amp and device.type == "cuda"))
    steps_per_epoch = math.ceil(len(dl_tr) / max(1, opt.acc_steps))
    total_steps = max(1, opt.epochs * steps_per_epoch)
    warm_iter = min(opt.warmup_epochs * steps_per_epoch, total_steps - 1)
    cosine_tot = max(1, total_steps - warm_iter)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optm,
        [
            torch.optim.lr_scheduler.LinearLR(optm, start_factor=0.1, end_factor=1.0, total_iters=warm_iter),
            torch.optim.lr_scheduler.CosineAnnealingLR(optm, T_max=cosine_tot, eta_min=1e-6),
        ],
        milestones=[warm_iter],
    )

    print(f"[Start] epochs={opt.epochs} bs={opt.bs} acc_steps={opt.acc_steps} sigma={opt.sigma}")

    global_step = 0
    last_metrics: Dict[str, float] = {}
    for ep in range(opt.epochs):
        net.train()
        if opt.freeze_den:
            net.den.eval()

        pbar = tqdm(dl_tr, desc=f"E{ep:03d}", leave=True)
        optm.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(pbar):
            a, p_im, n_im, m_im, has_sb, label = [x.to(device, non_blocking=True) for x in batch]
            has_sb = has_sb.bool()
            label = label.long()

            with autocast(enabled=(opt.amp and device.type == "cuda")):
                ea, ep_, en, em_, logits = net(
                    a,
                    p=p_im,
                    n=n_im,
                    m=m_im,
                    label=label,
                    denoise_no_grad=bool(opt.freeze_den),
                )

                ce_loss = F.cross_entropy(logits, label, label_smoothing=opt.label_smoothing)
                trip_loss = opt.trip_w * trip(ea, label)
                quad_loss = ea.new_tensor(0.0)
                if bool(has_sb.any()):
                    quad_loss = quad(ea, ep_, em_, en)

                loss = ce_loss + trip_loss + quad_loss

            loss_scaled = loss / max(1, opt.acc_steps)
            scaler.scale(loss_scaled).backward()

            if (batch_idx + 1) % opt.acc_steps == 0:
                scaler.step(optm)
                scaler.update()
                optm.zero_grad(set_to_none=True)
                scheduler.step()

            global_step += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

        if dl_va and (ep + 1) % opt.eval_int == 0:
            last_metrics = evaluate(dl_va, net, device, sigma=opt.sigma)
            print(
                f"[VAL] ep={ep} nn_top1={last_metrics['nn_top1']:.4f} auc={last_metrics['auc']:.4f} "
                f"same={last_metrics['same_mean']:.4f} diff={last_metrics['diff_mean']:.4f} var={last_metrics['var']:.4f} "
                f"| ams_ce={last_metrics['ams_ce']:.4f} ams_top1={last_metrics['ams_top1']:.4f} wrong_conf={last_metrics['wrong_conf']:.3f}"
            )

        if (ep + 1) % opt.save_int == 0:
            os.makedirs(opt.save_dir, exist_ok=True)
            tag = f"acc{last_metrics.get('ams_top1', 0.0):.4f}" if last_metrics else "accNA"
            ck = os.path.join(opt.save_dir, f"lcl_E{ep:03d}_{tag}.pth")
            torch.save(
                {
                    "epoch": ep,
                    "den": net.den.state_dict(),
                    "emb": net.emb.state_dict(),
                    "ams": net.ams.state_dict(),
                    "opt": optm.state_dict(),
                    "sched": scheduler.state_dict(),
                    "config": vars(opt),
                },
                ck,
            )
            print("[SAVE]", ck)


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Refactored train_img.py trainer.")
    ap.add_argument("--csv", default="./Path/train.csv")
    ap.add_argument("--csv_val", default="")
    ap.add_argument("--root", default="")

    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--acc_steps", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num_workers", type=int, default=10)
    ap.add_argument("--prefetch_factor", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--amp", action="store_true")

    ap.add_argument("--crop", type=int, default=0)
    ap.add_argument("--resize", type=int, default=0)

    ap.add_argument("--save_dir", default="ckps_lcl")
    ap.add_argument("--save_int", type=int, default=10)
    ap.add_argument("--eval_int", type=int, default=2)
    ap.add_argument("--pretrained", default="", help="checkpoint path/dir or 'auto'")
    ap.add_argument("--ffdnet_ckpt", default="./model/ckpts/ffdnet_gray.pth")

    ap.add_argument("--out_dim", type=int, default=256)
    ap.add_argument("--sigma", type=float, default=20 / 255.0)

    ap.add_argument("--m1", type=float, default=1.0)
    ap.add_argument("--m2", type=float, default=1.5)
    ap.add_argument("--lambda_m", type=float, default=2.0)
    ap.add_argument("--trip_margin", type=float, default=2.0)
    ap.add_argument("--trip_w", type=float, default=0.02)
    ap.add_argument("--label_smoothing", type=float, default=0.05)

    ap.add_argument("--lr_den", type=float, default=1e-4)
    ap.add_argument("--lr_emb", type=float, default=1e-4)
    ap.add_argument("--lr_ams", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-6)

    ap.add_argument("--ams_s", type=float, default=26.0)
    ap.add_argument("--ams_m", type=float, default=0.30)
    ap.add_argument("--warmup_epochs", type=int, default=2)
    return ap


def main() -> None:
    opt = build_argparser().parse_args()
    if not opt.csv:
        raise SystemExit("--csv is required")
    if opt.bs * opt.acc_steps < 32:
        warnings.warn("effective batch < 32; you may want to increase bs/acc_steps.")
    train(opt)


if __name__ == "__main__":
    main()
