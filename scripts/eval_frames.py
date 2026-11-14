"""在一批真实帧上统计模型数出的青蟹数量分布（整幅画面 vs 先裁出板子）。

用法： python scripts/eval_frames.py frames/1112 --start 113 --end 277 [--weights ...]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from board import find_board  # noqa: E402
from count import ROOT, class_names, load  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--end", type=int, default=10**9)
    ap.add_argument("--weights", default=str(ROOT / "runs/ft2/checkpoint_best_total.pth"))
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--dump", default=None, help="把每帧裁剪后的检测结果存成 json")
    a = ap.parse_args()
    model = load(a.weights)
    full, crop, nob = Counter(), Counter(), 0
    dump = {}
    for f in sorted(Path(a.dir).glob("f_*.jpg")):
        i = int(f.stem.split("_")[1])
        if not a.start <= i <= a.end:
            continue
        bgr = cv2.imread(str(f))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        n = class_names(model.predict(Image.fromarray(rgb), threshold=a.thr))
        full[n.count("green")] += 1
        box = find_board(bgr)
        if box is None:
            nob += 1
            continue
        x0, y0, x1, y1 = box
        det = model.predict(Image.fromarray(rgb[y0:y1, x0:x1]), threshold=a.thr)
        n = class_names(det)
        crop[(n.count("green"), len(n))] += 1
        dump[f.name] = {"box": box, "dets": [[*map(float, b), c, float(s)] for b, c, s in zip(det.xyxy, n, det.confidence)]}
    tot = sum(full.values())
    print(f"共 {tot} 帧，找不到板子 {nob} 帧")
    print("整幅画面  青蟹数分布:", dict(sorted(full.items())))
    g = Counter()
    for (gn, _), c in crop.items():
        g[gn] += c
    print("裁出板子  青蟹数分布:", dict(sorted(g.items())))
    print("裁出板子  (青蟹数, 总框数) 最常见:", crop.most_common(8))
    if a.dump:
        Path(a.dump).write_text(json.dumps(dump))


if __name__ == "__main__":
    main()
