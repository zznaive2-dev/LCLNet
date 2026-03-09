import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from model.LCLNet import AmsMarginProduct, ResNetEmbed
from model.ffdnet_network import FFDNet
from utils.lcl_eval_dataset import LCLTestDataset
from utils.lcl_eval_metrics import evaluate_lcl


def eval_lcl(args):
    """
    Eval for train_img / LCLNet-style checkpoints (den + emb + ams).
    """
    import pandas as pd
    from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

    device = torch.device(args.device if (args.device.startswith("cuda") and torch.cuda.is_available()) else "cpu")
    logging.info(f"[LCL] Using device: {device}")

    ds = LCLTestDataset(args.csv, root=args.root, crop=args.crop, resize=args.resize)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    den = FFDNet(1, 1, 64, 15, "R").to(device)
    emb = ResNetEmbed(out_dim=args.out_dim).to(device)
    ams = AmsMarginProduct(args.out_dim, len(ds.dev2cls), s=args.ams_s, m=args.ams_m).to(device)

    state = torch.load(args.ckpt, map_location=device)
    if not isinstance(state, dict):
        raise ValueError("Unsupported checkpoint format (expected dict with den/emb/ams).")
    if "den" in state:
        den.load_state_dict(state["den"], strict=False)
    if "emb" in state:
        emb.load_state_dict(state["emb"], strict=False)
    if "ams" in state:
        ams.load_state_dict(state["ams"], strict=False)


    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    exp_dir = out_dir / args.dir
    exp_dir.mkdir(parents=True, exist_ok=True)

    metrics = evaluate_lcl(
        loader,
        den=den,
        emb=emb,
        ams=ams,
        device=device,
        sigma_val=args.sigma,
        calc_ece=args.with_ece,
        T_eval=args.T,
    )

    def _sf(x, nd=4, na="n/a"):
        try:
            return f"{float(x):.{nd}f}"
        except Exception:
            return na

    print(
        f"[Eval] N={metrics['N']} | "
        f"NN-Top1={_sf(metrics['nn_top1'])}  "
        f"AUC={_sf(metrics['auc'])}  "
        f"AmsCE={_sf(metrics['ams_ce'])}  "
        f"AmsTop1={_sf(metrics['ams_top1'])}  "
        f"wrong_conf={_sf(metrics['wrong_conf'], 3)}  "
        f"ECE={_sf(metrics['ece'], 3)}  "
        f"wrong={metrics['num_wrong']}  "
        f"margin_wrong={_sf(metrics['margin_wrong'], 3)}"
    )

    # ROC curve (cosine sim)
    try:
        fpr = metrics["roc_fpr"]
        tpr = metrics["roc_tpr"]
        if len(fpr) and len(tpr):
            plt.figure(figsize=(6, 5))
            plt.plot(fpr, tpr, label=f"AUC = {metrics['auc']:.4f}")
            plt.plot([0, 1], [0, 1], "--", label="Chance")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC")
            plt.legend(loc="lower right")
            plt.tight_layout()
            roc_path = exp_dir / "roc_curve_cosine.png"
            plt.savefig(roc_path, dpi=200)
            plt.close()
            logging.info(f"[LCL] ROC saved: {roc_path}")
    except Exception as e:
        logging.warning(f"[LCL] ROC plot failed: {e}")

    # Softmax confusion matrix (device)
    y_true = metrics["y_true"]
    y_pred = metrics["y_pred"]
    class_names = metrics["class_names"]
    num_classes = len(class_names)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    prec, rec, f1, sup = precision_recall_fscore_support(y_true, y_pred, labels=list(range(num_classes)), zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    print(f"[F1-Softmax-Device] macro={macro_f1:.4f}  micro={micro_f1:.4f}  weighted={weighted_f1:.4f}")

    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(exp_dir / "confusion_matrix.csv", encoding="utf-8-sig")
    pd.DataFrame(
        {"device_id": class_names, "precision": prec, "recall": rec, "f1": f1, "support": sup}
    ).to_csv(exp_dir / "classification_report.csv", index=False, encoding="utf-8-sig")
    with open(exp_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write(f"macro_F1\t{macro_f1:.6f}\n")
        f.write(f"micro_F1\t{micro_f1:.6f}\n")
        f.write(f"weighted_F1\t{weighted_f1:.6f}\n")

    row_sums = cm.sum(axis=1, keepdims=True).astype(np.float64)
    cm_prob = np.divide(cm, np.maximum(row_sums, 1e-12), where=row_sums > 0)
    pd.DataFrame(cm_prob, index=class_names, columns=class_names).to_csv(exp_dir / "confusion_matrix_prob.csv", encoding="utf-8-sig")
    pd.DataFrame(cm_prob * 100.0, index=class_names, columns=class_names).to_csv(
        exp_dir / "confusion_matrix_percent.csv", encoding="utf-8-sig"
    )

    # NN-top1 confusion (device)
    y_pred_nn = metrics["y_pred_nn"]
    cm_nn = confusion_matrix(y_true, y_pred_nn, labels=list(range(num_classes)))
    pd.DataFrame(cm_nn, index=class_names, columns=class_names).to_csv(exp_dir / "nn_confusion_matrix.csv", encoding="utf-8-sig")

    # Model-level confusion (optional)
    if metrics.get("class_names_model"):
        y_true_m = metrics["y_true_model"]
        y_pred_m = metrics["y_pred_model"]
        class_names_m = metrics["class_names_model"]
        num_classes_m = len(class_names_m)
        cm_m = confusion_matrix(y_true_m, y_pred_m, labels=list(range(num_classes_m)))
        pd.DataFrame(cm_m, index=class_names_m, columns=class_names_m).to_csv(exp_dir / "model_confusion_matrix.csv", encoding="utf-8-sig")


def get_args():
    ap = argparse.ArgumentParser("Eval utilities (LCL only)")
    ap.add_argument("--csv", type=str, default="", help="test CSV with img_path, device_id")
    ap.add_argument("--root", type=str, default="", help="path prefix for img_path")
    ap.add_argument("--ckpt", type=str, default="", help="checkpoint (.pth) with den/emb/ams (or den/emb/arc)")
    ap.add_argument("--device", type=str, default="cuda", help="cuda/cpu")
    ap.add_argument("--out-dim", type=int, default=128, dest="out_dim", help="embedding dim")
    ap.add_argument("--ams-s", type=float, default=26.0, dest="ams_s", help="ams scale s")
    ap.add_argument("--ams-m", type=float, default=0.30, dest="ams_m", help="ams margin m (for CE)")
    ap.add_argument("--sigma", type=float, default=2 / 255.0, help="sigma for FFDNet")
    ap.add_argument("--T", type=float, default=1.4, help="temperature scaling for softmax")
    ap.add_argument("--with-ece", action="store_true", help="compute ECE")
    ap.add_argument("--crop", type=int, default=0, help="center crop size (0 disables)")
    ap.add_argument("--resize", type=int, default=0, help="resize (square), 0 disables")
    ap.add_argument("--batch-size", type=int, default=8, dest="batch_size", help="batch size")
    ap.add_argument("--num-workers", type=int, default=8, dest="num_workers", help="dataloader workers")
    ap.add_argument("--out-dir", type=str, default="./eval_results", help="output directory")
    ap.add_argument("--dir", type=str, default="/out", help="subfolder under out-dir")
    return ap.parse_args()


if __name__ == "__main__":
    args = get_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if not args.csv or not args.ckpt:
        raise SystemExit("Requires --csv and --ckpt")
    eval_lcl(args)
