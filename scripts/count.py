"""数图片里的欧洲青蟹（European Green crab）。

用法：
  python scripts/count.py 图片1.jpg [图片2.png ...]      # 输出数量，标框图存到 out/
  python scripts/count.py --eval                         # 在合成测试集上算计数准确率
选项： --weights runs/small/checkpoint_best_total.pth  --thr 0.5
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from rfdetr import RFDETRSmall

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["green", "rock", "jonah"]
COLORS = {"green": (0, 220, 0), "rock": (0, 140, 255), "jonah": (255, 0, 255)}


def load(weights: str) -> RFDETRSmall:
    model = RFDETRSmall(pretrain_weights=weights, resolution=512)
    try:
        model.optimize_for_inference()
    except Exception:  # 不是所有环境都支持，失败就用原模型
        pass
    return model


def class_names(det) -> list[str]:
    names = det.data.get("class_name") if hasattr(det, "data") else None
    if names is not None and len(names) == len(det):
        return [str(n) for n in names]
    return [CLASSES[int(c)] if 0 <= int(c) < len(CLASSES) else str(c) for c in det.class_id]


def detect(model, path: Path, thr: float):
    img = Image.open(path).convert("RGB")
    det = model.predict(img, threshold=thr)
    return img, det, class_names(det)


def draw(img: Image.Image, det, names: list[str], out: Path) -> None:
    bgr = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    for (x0, y0, x1, y1), n, s in zip(det.xyxy.astype(int), names, det.confidence):
        c = COLORS.get(n, (200, 200, 200))
        cv2.rectangle(bgr, (x0, y0), (x1, y1), c, 3 if n == "green" else 1)
        cv2.putText(bgr, f"{n} {s:.2f}", (x0, max(12, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1, cv2.LINE_AA)
    g = names.count("green")
    cv2.putText(bgr, f"Green crabs: {g}", (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(bgr, f"Green crabs: {g}", (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2, cv2.LINE_AA)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), bgr)


def evaluate(model, thr: float) -> None:
    split = ROOT / "dataset/test"
    coco = json.loads((split / "_annotations.coco.json").read_text())
    gt = Counter(a["image_id"] for a in coco["annotations"] if a["category_id"] == 1)
    ok = 0
    errs = []
    for im in coco["images"]:
        _, _, names = detect(model, split / im["file_name"], thr)
        pred, true = names.count("green"), gt.get(im["id"], 0)
        ok += pred == true
        if pred != true:
            errs.append((im["file_name"], true, pred))
    n = len(coco["images"])
    print(f"合成测试集：{ok}/{n} 张青蟹数量完全正确（{ok / n:.1%}），阈值 {thr}")
    for f, t, p in errs[:15]:
        print(f"  {f}: 真实 {t}，预测 {p}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="*")
    ap.add_argument("--weights", default=str(ROOT / "runs/small/checkpoint_best_total.pth"))
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "out"))
    a = ap.parse_args()
    if not a.images and not a.eval:
        ap.error("给图片路径，或者加 --eval")
    model = load(a.weights)
    if a.eval:
        evaluate(model, a.thr)
    for p in map(Path, a.images):
        img, det, names = detect(model, p, a.thr)
        c = Counter(names)
        print(f"{p.name}: 青蟹 {c['green']} 只（rock {c['rock']}，jonah {c['jonah']}）")
        draw(img, det, names, Path(a.out) / f"{p.stem}_count.jpg")


if __name__ == "__main__":
    sys.exit(main())
