"""颜色退化增强：模拟「红色被水吸收、偏蓝青、发灰」的水下光线。

2026-11-15 新场地 + 新摄像头下，rock / jonah 的红色完全拍不出来，模型靠颜色分品种就全错了。
这里把已有数据（旧场地真实帧、合成图）随机去红、加青色偏色、降饱和、降对比度（一部分直接变灰度），
逼模型靠形状和花纹分品种。框不变，只改颜色。

用法： python scripts/color_aug.py --src real/pool1112 real/pool1112_cp --name pool1112_dark
       python scripts/color_aug.py --src dataset --name synth_dark --sample 600
输出： real/<name>/{train,valid}（没有 _log.json，make_ft_dataset 会全部使用）
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def degrade(bgr: np.ndarray, rng: random.Random) -> np.ndarray:
    img = bgr.astype(np.float32)
    if rng.random() < 0.2:
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        img = np.stack([g, g, g], -1)
    else:
        img[..., 2] *= rng.uniform(0.05, 0.5)              # 去红（BGR 的 R）
        img[..., 1] *= rng.uniform(0.85, 1.05)
        img[..., 0] *= rng.uniform(0.95, 1.15)
        gray = img.mean(-1, keepdims=True)
        img = gray + (img - gray) * rng.uniform(0.3, 1.0)  # 降饱和
        img += np.float32([rng.uniform(0, 40), rng.uniform(0, 35), 0])  # 青色偏色（B、G 抬高）
    mean = img.mean()
    img = mean + (img - mean) * rng.uniform(0.55, 0.95)    # 降对比度
    img *= rng.uniform(0.75, 1.1)
    out = np.clip(img, 0, 255).astype(np.uint8)
    if rng.random() < 0.3:
        out = cv2.GaussianBlur(out, (0, 0), rng.uniform(0.6, 1.4))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", nargs="+", required=True, help="COCO 数据集目录（含 train/valid）")
    ap.add_argument("--name", required=True)
    ap.add_argument("--sample", type=int, default=0, help="每个来源 train 最多取几张（0=全部）；valid 取 1/10")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    out = ROOT / "real" / a.name
    if out.exists():
        shutil.rmtree(out)
    for split in ("train", "valid"):
        d = out / split
        d.mkdir(parents=True)
        res = {"images": [], "annotations": [], "categories": None}
        for src in a.src:
            sdir = ROOT / src / split
            coco = json.loads((sdir / "_annotations.coco.json").read_text())
            res["categories"] = res["categories"] or coco["categories"]
            imgs = coco["images"]
            if a.sample:
                k = a.sample if split == "train" else max(1, a.sample // 10)
                imgs = rng.sample(imgs, min(k, len(imgs)))
            anns = {}
            for an in coco["annotations"]:
                anns.setdefault(an["image_id"], []).append(an)
            tag = Path(src).name
            for img in imgs:
                bgr = cv2.imread(str(sdir / img["file_name"]))
                if bgr is None:
                    continue
                iid = len(res["images"])
                name = f"{a.name}_{tag}_{img['file_name']}"
                cv2.imwrite(str(d / name), degrade(bgr, rng), [cv2.IMWRITE_JPEG_QUALITY, 92])
                res["images"].append({**img, "id": iid, "file_name": name})
                for an in anns.get(img["id"], []):
                    res["annotations"].append({**an, "id": len(res["annotations"]), "image_id": iid})
        (d / "_annotations.coco.json").write_text(json.dumps(res))
        print(f"{split}: {len(res['images'])} 张图, {len(res['annotations'])} 个框")


if __name__ == "__main__":
    main()
