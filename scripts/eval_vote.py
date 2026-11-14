"""在一段录像上比较「单帧计数」和「多帧投票」的正确率。

用法： python scripts/eval_vote.py videos/pool_2026-11-12.mkv --start 113 --truth 4 [--weights ...] [--holdout 247]
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import cv2
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from count import ROOT, detect_image, load, vote  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--start", type=float, default=0)
    ap.add_argument("--truth", type=int, required=True)
    ap.add_argument("--weights", default=str(ROOT / "runs/ft2/checkpoint_best_total.pth"))
    ap.add_argument("--holdout", type=float, default=None, help="这一秒之后的画面没参与训练，单独统计")
    ap.add_argument("--fps", type=float, default=5)
    ap.add_argument("--window", type=float, default=3)
    a = ap.parse_args()
    model = load(a.weights)
    cap = cv2.VideoCapture(a.video)
    cap.set(cv2.CAP_PROP_POS_MSEC, a.start * 1000)
    native = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = round(native / a.fps)
    samples, i = [], 0  # (时间, 青蟹数)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            _, _, names = detect_image(model, Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), 0.5)
            samples.append((a.start + i / native, names.count("green")))
        i += 1
    n = int(a.window * a.fps)

    def report(label, ss):
        if len(ss) < n:
            return
        single = sum(c == a.truth for _, c in ss) / len(ss)
        wins = [vote([c for _, c in ss[j:j + n]]) for j in range(0, len(ss) - n + 1, int(a.fps))]
        ok = sum(w == a.truth for w in wins)
        print(f"{label}: 单帧 {single:.1%}（{len(ss)} 帧，分布 {dict(sorted(Counter(c for _, c in ss).items()))}）；"
              f"{a.window:g} 秒投票 {ok}/{len(wins)} 个窗口正确（{ok / len(wins):.1%}），错的窗口结果 {dict(Counter(w for w in wins if w != a.truth))}")

    report("整段", samples)
    if a.holdout is not None:
        report(f"{a.holdout:g} 秒之后（没参与训练）", [s for s in samples if s[0] >= a.holdout])


if __name__ == "__main__":
    main()
