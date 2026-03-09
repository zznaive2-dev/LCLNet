from __future__ import annotations

import os
import re
from typing import Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class LCLTestDataset(Dataset):
    """
    Evaluation dataset for train_img-style LCL pipeline.

    CSV must include columns:
      - img_path: relative or absolute path to .npy (preferred) or image file
      - device_id: device label

    Additionally parses a "model" (brand) from img_path using MODEL_RE for model-level evaluation.
    """

    MODEL_RE = re.compile(r"(Apple|Huawei|Xiaomi|OPPO|VIVO|Samsung|Sony|LG|Lenovo|OnePlus|Asus|Wiko|Microsoft)",re.I)

    def __init__(self, csv_path: str, root: str = "", crop: int = 0, resize: int = 0):
        self.df = pd.read_csv(csv_path).reset_index(drop=True)
        if not {"img_path", "device_id"}.issubset(self.df.columns):
            raise ValueError("CSV must include columns: img_path, device_id")

        self.root = root.rstrip("/\\")
        self.crop = int(crop) if crop else 0
        self.resize = (int(resize), int(resize)) if resize else None

        uniq_dev = sorted(self.df["device_id"].unique().tolist())
        self.dev2cls: Dict[str, int] = {d: i for i, d in enumerate(uniq_dev)}
        self.id2dev: Dict[int, str] = {cls_idx: str(dev_val) for dev_val, cls_idx in self.dev2cls.items()}

        self.df["model"] = self.df["img_path"].apply(self._parse_model)
        uniq_model = sorted(self.df["model"].unique().tolist())
        self.model2cls: Dict[str, int] = {m: i for i, m in enumerate(uniq_model)}
        self.id2model: Dict[int, str] = {i: m for m, i in self.model2cls.items()}

        self.devcls2modelcls: Dict[int, int] = {}
        for dev, dev_idx in self.dev2cls.items():
            m = self.df[self.df.device_id == dev].iloc[0].model
            self.devcls2modelcls[int(dev_idx)] = int(self.model2cls[m])

    def _parse_model(self, path: str) -> str:
        m = self.MODEL_RE.search(str(path))
        return m.group(1).upper() if m else "UNK"

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_path(self, p: str) -> str:
        p = str(p)
        if os.path.isabs(p):
            return p
        if self.root:
            return os.path.join(self.root, p)
        return p

    def _imread_npy_or_img(self, p: str) -> torch.Tensor:
        path = self._resolve_path(p)
        if path.lower().endswith(".npy"):
            arr = np.load(path, mmap_mode="r")
            if (not arr.flags.writeable) or isinstance(arr, np.memmap):
                arr = np.array(arr, copy=True)
            t = torch.from_numpy(arr).to(torch.float32)
            if float(t.max()) > 1.5:
                t = t / 255.0
            t = t.clamp(0.0, 1.0)
            return t.unsqueeze(0)  # [1,H,W]

        # image file path (optional dependency: cv2)
        try:
            import cv2  # type: ignore
        except Exception as e:
            raise RuntimeError("cv2 is required to evaluate non-npy image files.") from e

        arr = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if arr is None:
            raise FileNotFoundError(path)

        if self.crop:
            h, w = arr.shape[:2]
            y0, x0 = (h - self.crop) // 2, (w - self.crop) // 2
            arr = arr[y0 : y0 + self.crop, x0 : x0 + self.crop]
        if self.resize:
            arr = cv2.resize(arr, self.resize[::-1], interpolation=cv2.INTER_AREA)
        if arr.ndim == 3:
            arr = arr.astype(np.float32)
            arr = (0.29893602 * arr[..., 2] + 0.58704307 * arr[..., 1] + 0.11402090 * arr[..., 0])
        arr = arr.astype(np.float32)
        t = torch.from_numpy(arr)
        if float(t.max()) > 1.5:
            t = t / 255.0
        t = t.clamp(0.0, 1.0)
        return t.unsqueeze(0)

    def __getitem__(self, idx: int):
        r = self.df.iloc[idx]
        x = self._imread_npy_or_img(r.img_path)
        y_dev = torch.tensor(self.dev2cls[r.device_id], dtype=torch.long)
        y_model = torch.tensor(self.model2cls[r.model], dtype=torch.long)
        return x, y_dev, y_model, str(r.img_path)

