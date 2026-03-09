#!/usr/bin/env python3
import argparse
import csv
import os
from multiprocessing.pool import ThreadPool

import cv2
import numpy as np
from tqdm import tqdm

RGB2GRAY = np.array([0.11402090, 0.58704307, 0.29893602], np.float32).reshape(3, 1)

CSV_DEFAULT = "./Path/img_csv_path"  # CSV file path (must contain an img_path column)
ROOT_DEFAULT = "../data"  # Root directory for original images (prefix for relative img_path)


def rgb2gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img.astype(np.float32)
    if img.shape[2] == 3:
        h, w = img.shape[:2]
        return (img.reshape(-1, 3) @ RGB2GRAY).reshape(h, w).astype(np.float32)
    if img.shape[2] == 4:
        return rgb2gray(img[:, :, :3])
    raise ValueError("Unsupported channel number")


def convert_one(rel_path: str, root: str, out_root: str, crop: int, resize: int):
    src = rel_path if (not root) or os.path.isabs(rel_path) else os.path.join(root, rel_path)
    img = cv2.imdecode(np.fromfile(src, np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        return f"[WARN] unreadable: {src}"

    # Rotation rule: if height > width, rotate 90 degrees CCW once.
    h, w = img.shape[:2]
    if h > w:
        img = np.rot90(img, 1, axes=(0, 1)).copy()

    # 1) Optional center crop
    if crop:
        h, w = img.shape[:2]
        y0 = (h - crop) // 2
        x0 = (w - crop) // 2
        img = img[y0 : y0 + crop, x0 : x0 + crop]

    # 2) Optional resize
    if resize:
        img = cv2.resize(img, (resize, resize), interpolation=cv2.INTER_AREA)

    # 3) Convert to grayscale float32
    img = rgb2gray(img)

    # 4) Save .npy (mirrors input relative directory structure; strips leading data/)
    rel_norm = rel_path.replace("\\", "/").lstrip("./")
    if rel_norm.startswith("../"):
        rel_norm = rel_norm[3:]
    if rel_norm.startswith("data/"):
        rel_norm = rel_norm[len("data/") :]
    dest = os.path.join(out_root, os.path.splitext(rel_norm)[0] + ".npy")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    np.save(dest, img.astype(np.float32))
    return None


def main(args) -> None:
    rel_list = []
    with open(args.csv, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rel_list.extend(row["img_path"] for row in reader)

    print(f"Total images: {len(rel_list)}")
    pool = ThreadPool(args.workers)
    errs = []
    for err in tqdm(
        pool.imap_unordered(lambda p: convert_one(p, args.root, args.out_dir, args.crop, args.resize), rel_list),
        total=len(rel_list),
        unit="img",
    ):
        if err:
            errs.append(err)
    pool.close()
    pool.join()

    if errs:
        print(f"\nCompleted with {len(errs)} warnings:")
        for e in errs[:30]:
            print("  ", e)
    else:
        print("\nAll images converted successfully.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=CSV_DEFAULT, help="CSV file containing img_path column")
    ap.add_argument("--root", default=ROOT_DEFAULT, help="Root dir prefix for relative img_path in CSV")
    ap.add_argument("--out_dir", default="", help="Output root directory for .npy files")
    ap.add_argument("--crop", type=int, default=512, help="Center crop size (0 = disable)")
    ap.add_argument("--resize", type=int, default=0, help="Final resize side length (0 = disable)")
    ap.add_argument("--workers", type=int, default=16, help="Number of worker threads")
    main(ap.parse_args())
