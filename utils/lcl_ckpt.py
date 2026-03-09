from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch


def find_latest_ckpt(path: str) -> Optional[str]:
    if not os.path.isdir(path):
        return None
    cand = [f for f in os.listdir(path) if f.endswith(".pth")]
    if not cand:
        return None

    def _ep(f: str) -> int:
        m = re.search(r"_E(\\d+)\\.pth$", f)
        return int(m.group(1)) if m else -1

    cand.sort(key=_ep, reverse=True)
    return os.path.join(path, cand[0])


def strip_module_prefix(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    return {k.replace("module.", ""): v for k, v in state_dict.items()}


def normalize_state_dict(obj: Any) -> Dict[str, Any]:
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        obj = obj["state_dict"]
    if not isinstance(obj, dict):
        raise TypeError("Unsupported checkpoint format (expected dict/state_dict).")
    return strip_module_prefix(obj)


@dataclass(frozen=True)
class LoadInfo:
    missing_keys: Tuple[str, ...]
    unexpected_keys: Tuple[str, ...]


def load_weights(module: torch.nn.Module, ckpt_path: str, strict: bool = False) -> LoadInfo:
    sd = torch.load(ckpt_path, map_location="cpu")
    sd = normalize_state_dict(sd)
    info = module.load_state_dict(sd, strict=strict)

    try:
        missing = tuple(info.missing_keys)
        unexpected = tuple(info.unexpected_keys)
    except AttributeError:
        try:
            missing, unexpected = info  # type: ignore[misc]
            missing = tuple(missing)
            unexpected = tuple(unexpected)
        except Exception:
            missing, unexpected = tuple(), tuple()
    return LoadInfo(missing_keys=missing, unexpected_keys=unexpected)

