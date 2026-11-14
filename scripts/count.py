"""数图片里的欧洲青蟹（European Green crab）。

用法：
  python scripts/count.py 图片1.jpg [图片2.png ...]      # 输出数量，标框图存到 out/
  python scripts/count.py --video 录像.mkv --at 200      # 从第 200 秒起取 3 秒画面，多帧投票
  python scripts/count.py --video rtsp://192.168.1.10:554/stream1   # 直接从 ROV 视频流取 3 秒投票
  python scripts/count.py --eval [数据集目录]             # 在测试集上算计数准确率（默认 dataset/test）
选项： --weights <权重>  --thr 0.5  --no-crop（不自动裁出板子）

流程：先在画面里找到白色样本板并裁出来（广角画面里螃蟹太小），再识别；
同一位置重叠的框只保留置信度最高的一个。
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from board import find_board  # noqa: E402

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


def dedup(det, iou_thr: float = 0.7):
    """不分种类的去重：两个框重叠很多时只留置信度高的。"""
    order = np.argsort(-det.confidence)
    keep: list[int] = []
    for i in order:
        a = det.xyxy[i]
        ok = True
        for j in keep:
            b = det.xyxy[j]
            iw = min(a[2], b[2]) - max(a[0], b[0]); ih = min(a[3], b[3]) - max(a[1], b[1])
            if iw > 0 and ih > 0:
                inter = iw * ih
                union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
                if inter / union > iou_thr:
                    ok = False
                    break
        if ok:
            keep.append(int(i))
    return det[sorted(keep)]


def detect(model, path: Path, thr: float, crop: bool = True):
    return detect_image(model, Image.open(path).convert("RGB"), thr, crop)


def detect_image(model, img: Image.Image, thr: float, crop: bool = True):
    sub, off = img, (0, 0)
    if crop:
        box = find_board(cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR))
        if box is not None:
            sub, off = img.crop(box), box[:2]
    det = dedup(model.predict(sub, threshold=thr))
    det.xyxy = det.xyxy + np.array([off[0], off[1], off[0], off[1]], dtype=det.xyxy.dtype)
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


def vote(counts: list[int]) -> int:
    """多帧投票：取出现最多的数；票数相同取大的（单帧出错基本都是被挡住而少数）。"""
    c = Counter(counts)
    top = max(c.values())
    return max(k for k, v in c.items() if v == top)


def count_video(model, src: str, thr: float, at: float = 0.0, seconds: float = 3.0, fps: float = 5.0, crop: bool = True):
    """从录像或视频流里取一小段，每帧各数一次再投票。返回 (结果, 各帧计数, 代表帧的 (img, det, names))。"""
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"打不开视频：{src}")
    live = "://" in src
    if not live and at > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, at * 1000)
    native = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(native / fps))
    total = int(seconds * native)
    results = []
    for i in range(total):
        ok, frame = cap.read()
        if not ok:
            break
        if i % step:
            continue
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        _, det, names = detect_image(model, img, thr, crop)
        results.append((names.count("green"), img, det, names))
    cap.release()
    if not results:
        raise SystemExit("没读到画面")
    counts = [r[0] for r in results]
    best = vote(counts)
    rep = max((r for r in results if r[0] == best), key=lambda r: float(r[2].confidence.min()) if len(r[2]) else 0.0)
    return best, counts, rep[1:]


def evaluate(model, thr: float, split: Path) -> None:
    coco = json.loads((split / "_annotations.coco.json").read_text())
    gt = Counter(a["image_id"] for a in coco["annotations"] if a["category_id"] == 1)
    ok = 0
    errs = []
    for im in coco["images"]:
        _, _, names = detect(model, split / im["file_name"], thr, crop=False)
        pred, true = names.count("green"), gt.get(im["id"], 0)
        ok += pred == true
        if pred != true:
            errs.append((im["file_name"], true, pred))
    n = len(coco["images"])
    print(f"{split}：{ok}/{n} 张青蟹数量完全正确（{ok / n:.1%}），阈值 {thr}")
    for f, t, p in errs[:15]:
        print(f"  {f}: 真实 {t}，预测 {p}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="*")
    ap.add_argument("--weights", default=str(ROOT / "runs/ft2/checkpoint_best_total.pth"))
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--eval", nargs="?", const=str(ROOT / "dataset/test"), default=None)
    ap.add_argument("--no-crop", action="store_true")
    ap.add_argument("--video", default=None, help="录像文件或视频流地址，多帧投票")
    ap.add_argument("--at", type=float, default=0.0, help="录像从第几秒开始取")
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--fps", type=float, default=5.0)
    ap.add_argument("--out", default=str(ROOT / "out"))
    a = ap.parse_args()
    if not a.images and not a.eval and not a.video:
        ap.error("给图片路径，或者加 --video / --eval")
    model = load(a.weights)
    if a.eval:
        evaluate(model, a.thr, Path(a.eval))
    if a.video:
        best, counts, (img, det, names) = count_video(model, a.video, a.thr, a.at, a.seconds, a.fps, not a.no_crop)
        dist = ", ".join(f"{k} 只 x{v} 帧" for k, v in sorted(Counter(counts).items()))
        print(f"青蟹 {best} 只（{len(counts)} 帧投票：{dist}）")
        draw(img, det, names, Path(a.out) / "video_count.jpg")
    for p in map(Path, a.images):
        img, det, names = detect(model, p, a.thr, crop=not a.no_crop)
        c = Counter(names)
        print(f"{p.name}: 青蟹 {c['green']} 只（rock {c['rock']}，jonah {c['jonah']}）")
        draw(img, det, names, Path(a.out) / f"{p.stem}_count.jpg")


if __name__ == "__main__":
    sys.exit(main())
