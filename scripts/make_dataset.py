"""生成 RF-DETR 用的 COCO 数据集：dataset/{train,valid,test}/_annotations.coco.json

用法： python scripts/make_dataset.py --train 4000 --valid 500 --test 300
       python scripts/make_dataset.py --preview 12   # 只出几张带框预览图看效果
"""
from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import synth  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "dataset"
COLORS = [(0, 220, 0), (0, 120, 255), (255, 0, 255)]  # green / rock / jonah

_CRABS = None


def _init():
    global _CRABS
    cv2.setNumThreads(1)  # 每个进程只用 1 个线程，避免 workers x OpenCV 线程把 CPU 吃满
    _CRABS = synth.load_crabs()


def _job(args):
    split, idx, seed = args
    rng = np.random.default_rng(seed)
    img, anns = synth.render(rng, _CRABS)
    name = f"{split}_{idx:05d}.jpg"
    cv2.imwrite(str(OUT / split / name), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return name, img.shape[1], img.shape[0], anns


def build_split(split: str, n: int, base_seed: int, workers: int) -> None:
    (OUT / split).mkdir(parents=True, exist_ok=True)
    jobs = [(split, i, base_seed * 1_000_003 + i) for i in range(n)]
    images, annotations = [], []
    with Pool(workers, initializer=_init) as pool:
        for k, (name, w, h, anns) in enumerate(pool.imap(_job, jobs, chunksize=8)):
            images.append({"id": k, "file_name": name, "width": w, "height": h})
            for cls, x, y, bw, bh in anns:
                annotations.append({
                    "id": len(annotations), "image_id": k, "category_id": cls + 1,
                    "bbox": [round(x, 2), round(y, 2), round(bw, 2), round(bh, 2)],
                    "area": round(bw * bh, 2), "iscrowd": 0,
                })
            if (k + 1) % 500 == 0:
                print(f"  {split}: {k + 1}/{n}", flush=True)
    coco = {
        "info": {"description": "MATE 2026 NAVIGATOR Task 2.1 synthetic crab boards"},
        "images": images,
        "annotations": annotations,
        "categories": [{"id": i + 1, "name": c, "supercategory": "crab"} for i, c in enumerate(synth.CLASSES)],
    }
    (OUT / split / "_annotations.coco.json").write_text(json.dumps(coco))
    per = np.bincount([a["category_id"] for a in annotations], minlength=4)[1:]
    print(f"{split}: {n} 张图, {len(annotations)} 个框, 各类 {dict(zip(synth.CLASSES, per.tolist()))}")


def preview(n: int) -> None:
    _init()
    d = ROOT / "preview"; d.mkdir(exist_ok=True)
    for i in range(n):
        rng = np.random.default_rng(99_000 + i)
        img, anns = synth.render(rng, _CRABS)
        for cls, x, y, w, h in anns:
            cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), COLORS[cls], 2)
            cv2.putText(img, synth.CLASSES[cls], (int(x), int(y) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[cls], 1)
        g = sum(1 for a in anns if a[0] == 0)
        cv2.putText(img, f"green={g}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imwrite(str(d / f"preview_{i:02d}.jpg"), img)
    print("预览图写到", d)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=4000)
    ap.add_argument("--valid", type=int, default=500)
    ap.add_argument("--test", type=int, default=300)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--preview", type=int, default=0)
    a = ap.parse_args()
    if a.preview:
        preview(a.preview)
    else:
        for split, n, seed in [("train", a.train, 1), ("valid", a.valid, 2), ("test", a.test, 3)]:
            build_split(split, n, seed, a.workers)
