from __future__ import annotations

import os
import random
import re
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


MODEL_RE = re.compile(
    r"(Apple|Huawei|Xiaomi|OPPO|VIVO|Samsung|Sony|LG|Lenovo|OnePlus|Asus|Wiko|Microsoft)",
    re.I,
)


def _to_float01(arr: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(arr).to(torch.float32)
    if arr.dtype.kind in ("u", "i"):
        t = t / 255.0
    else:
        if float(t.max() > 1.5):
            t = t / 255.0
    return t.clamp(0, 1)


class QuadDataset(Dataset):
    """
    Returns a 4-tuple of grayscale noise patches for quadruplet training:
      (anchor, positive_same_device, negative_diff_model, negative_same_model_diff_device)

    Output tuple:
      a, p, n, m, has_sb, label
    where has_sb indicates if "same model but different device" exists.
    """

    def __init__(self, csv_path: str, root: str = "", crop: int = 0, resize: int = 0):
        self.df = pd.read_csv(csv_path).reset_index(drop=True)
        if not {"img_path", "device_id"}.issubset(self.df.columns):
            raise ValueError("CSV must include columns: img_path, device_id")

        self.root = root.rstrip("/\\")
        self.crop = int(crop) if crop else 0
        self.resize = (int(resize), int(resize)) if resize else None

        self.df["model"] = self.df.img_path.apply(
            lambda p: MODEL_RE.search(str(p)).group(1).upper() if MODEL_RE.search(str(p)) else "UNK"
        )

        uniq_dev = sorted(self.df.device_id.unique().tolist())
        self.dev2cls = {d: i for i, d in enumerate(uniq_dev)}
        self.cls_num = len(uniq_dev)

        self.grp_dev = self.df.groupby("device_id").groups
        self.grp_model = self.df.groupby("model").groups

        self.hard_neg = {}
        self.hn_ready = False

    def _resolve_path(self, rel: str) -> str:
        rel = str(rel)
        if self.root:
            return os.path.join(self.root, rel)
        return rel

    def _load_img(self, rel_path: str) -> torch.Tensor:
        path = self._resolve_path(rel_path)
        arr = np.load(path, mmap_mode="r")
        if (not arr.flags.writeable) or isinstance(arr, np.memmap):
            arr = np.array(arr, copy=True)

        t = _to_float01(arr)
        if t.ndim == 2:
            t = t.unsqueeze(0)  # [1,H,W]
        elif t.ndim == 3:
            # accept HWC or CHW and convert to [1,H,W] by luma
            if t.shape[0] in (1, 3) and t.shape[-1] not in (1, 3):
                t = t  # CHW already
            else:
                t = t.permute(2, 0, 1)  # HWC -> CHW
            if t.shape[0] == 3:
                t = t.mean(dim=0, keepdim=True)
            elif t.shape[0] != 1:
                raise ValueError(f"Unsupported channel count in npy: {t.shape} ({path})")
        else:
            raise ValueError(f"Unsupported npy shape: {t.shape} ({path})")

        # NOTE: original train_img.py didn't actually apply crop/resize for npy inputs.
        return t.contiguous()

    @staticmethod
    def _rand_except(idxs, ex: int) -> int:
        pool = [i for i in idxs if i != ex]
        if not pool:
            return ex
        return random.choice(pool)

    def _sample_quad(self, idx: int) -> Tuple[int, int, int, int, bool, int]:
        ra = self.df.iloc[idx]
        dev = ra.device_id
        md = ra.model

        rp = self.df.iloc[self._rand_except(self.grp_dev[dev], idx)]

        if self.hn_ready and idx in self.hard_neg:
            rn = self.df.iloc[self.hard_neg[idx]]
        else:
            other_models = [m for m in self.grp_model if m != md]
            if other_models:
                rn_md = random.choice(other_models)
                rn = self.df.iloc[random.choice(self.grp_model[rn_md])]
            else:
                rn = self.df.iloc[self._rand_except(self.grp_model[md], idx)]

        sb = [i for i in self.grp_model.get(md, []) if self.df.iloc[i].device_id != dev]
        has_sb = len(sb) > 0
        rm = self.df.iloc[random.choice(sb)] if has_sb else rn

        label = int(self.dev2cls[dev])
        return int(ra.name), int(rp.name), int(rn.name), int(rm.name), bool(has_sb), label

    def __getitem__(self, idx: int):
        i_a, i_p, i_n, i_m, has_sb, label = self._sample_quad(idx)
        ra = self.df.iloc[i_a]
        rp = self.df.iloc[i_p]
        rn = self.df.iloc[i_n]
        rm = self.df.iloc[i_m]

        a = self._load_img(ra.img_path)
        p = self._load_img(rp.img_path)
        n = self._load_img(rn.img_path)
        m = self._load_img(rm.img_path)

        return (
            a,
            p,
            n,
            m,
            torch.tensor(has_sb, dtype=torch.bool),
            torch.tensor(label, dtype=torch.long),
        )

    def __len__(self) -> int:
        return len(self.df)

