"""真实螃蟹「剪贴」增强：把真实画面里的螃蟹换位置、换角度、换数量，打破固定摆法。

做法（按板面几何来，透视是对的）：
  1. 每帧找到板子四角 -> 单应矩阵 H（画面 -> 板面俯视坐标）
  2. 把每只螃蟹从画面「抬」到板面上，存成俯视小图（种类已知）
  3. 把原画面里的螃蟹抹掉，得到空板子
  4. 在板面上随机选位置和朝向，把俯视小图按目标帧的 H 投影回去贴上

用法： python scripts/copypaste.py --name pool1112 --per-frame 5 [--preview 9]
输出： real/<name>_cp/{train,valid}（valid 只用 valid 帧里的螃蟹，不和训练集串）
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from board import find_quad  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
N = 800
SQUARE = np.float32([[0, 0], [N, 0], [N, N], [0, N]])
CLASSES = ["green", "rock", "jonah"]
COLORS = [(0, 220, 0), (0, 140, 255), (255, 0, 255)]


def ellipse_pts(x, y, w, h, scale=1.0, n=40):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([x + w / 2 + scale * w / 2 * np.cos(t), y + h / 2 + scale * h / 2 * np.sin(t)], 1).astype(np.float32)


def board_lum(img, quad):
    m = np.zeros(img.shape[:2], np.uint8)
    cv2.fillPoly(m, [quad.astype(np.int32)], 255)
    v = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., 2]
    return float(np.median(v[m > 0]))


def lift(img, H, box):
    """把一只螃蟹抬到板面：返回 (俯视小图, alpha, 板面半径 r, 每板面单位像素数 k)。"""
    x, y, w, h = box
    bp = cv2.perspectiveTransform(ellipse_pts(x, y, w, h)[None], H)[0]
    c = bp.mean(0)
    r = float(np.linalg.norm(bp - c, axis=1).max()) * 1.05
    area_b = cv2.contourArea(bp)
    if area_b < 50 or r > N / 2:
        return None
    k = math.sqrt((math.pi / 4 * w * h) / area_b)
    S = int(math.ceil(2 * r * k))
    if S < 16 or S > 500:
        return None
    K = np.array([[k, 0, (r - c[0]) * k], [0, k, (r - c[1]) * k], [0, 0, 1]], np.float64)
    patch = cv2.warpPerspective(img, K @ H, (S, S), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    alpha = np.zeros((S, S), np.float32)
    cv2.fillPoly(alpha, [((bp - c + r) * k).astype(np.int32)], 1.0)
    alpha = cv2.GaussianBlur(alpha, (0, 0), max(1.0, 0.035 * S))
    return patch, alpha, r, k


def gripper_mask(img):
    """ROV 爪子：从画面下边缘伸进来的大块黑色。"""
    h, w = img.shape[:2]
    dark = (cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., 2] < 60).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(dark)
    grip = np.zeros((h, w), np.uint8)
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if y + bh >= h - 2 and area > 0.002 * h * w:
            grip[lab == i] = 1
    return cv2.dilate(grip, np.ones((9, 9), np.uint8)) > 0


def erase(img, boxes, quad, grip):
    """抹掉螃蟹：只在板子范围内、避开爪子，用周围板子本身的颜色平滑填上。"""
    h, w = img.shape[:2]
    rects = np.zeros((h, w), np.uint8)
    for x, y, bw, bh in boxes:
        px, py = 0.14 * bw + 3, 0.14 * bh + 3
        cv2.rectangle(rects, (int(x - px), int(y - py)), (int(x + bw + px), int(y + bh + py)), 1, -1)
    board = np.zeros((h, w), np.uint8)
    cv2.fillPoly(board, [quad.astype(np.int32)], 1)
    board = cv2.dilate(board, np.ones((7, 7), np.uint8))
    m = ((rects > 0) & (board > 0) & ~grip).astype(np.float32)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    boardlike = (hsv[..., 1] < 90) & (hsv[..., 2] > 120)
    keep = ((board > 0) & (rects == 0) & ~grip & boardlike).astype(np.float32)
    sig = max(8.0, 0.035 * max(h, w))
    num = cv2.GaussianBlur(img.astype(np.float32) * keep[..., None], (0, 0), sig)
    den = cv2.GaussianBlur(keep, (0, 0), sig)[..., None]
    fill = np.where(den > 1e-3, num / np.maximum(den, 1e-3), np.float32(board_lum(img, quad)))
    fill = fill + np.random.default_rng(0).normal(0, 2.0, fill.shape).astype(np.float32)
    soft = np.minimum(cv2.GaussianBlur(m, (0, 0), 2.0), m + 0.0)[..., None]
    soft = cv2.GaussianBlur(m, (0, 0), 1.5)[..., None] * m[..., None]
    return np.clip(img * (1 - soft) + fill * soft, 0, 255).astype(np.float32)


def load_split(base: Path, split: str, clean: set[str] | None):
    coco = json.loads((base / split / "_annotations.coco.json").read_text())
    anns: dict[int, list] = {}
    for a in coco["annotations"]:
        anns.setdefault(a["image_id"], []).append((a["category_id"] - 1, a["bbox"]))
    out = []
    for im in coco["images"]:
        if clean is not None and im["file_name"] not in clean:
            continue
        out.append((base / split / im["file_name"], anns.get(im["id"], [])))
    return out


def build(frames, rng, per_frame: int, lib_stride: int):
    """frames: [(path, [(cls, box)])] -> 生成 [(img, anns)]。"""
    prepared, lib = [], {0: [], 1: [], 2: []}
    for i, (path, anns) in enumerate(frames):
        img = cv2.imread(str(path))
        quad = find_quad(img)
        if quad is None:
            continue
        H = cv2.getPerspectiveTransform(quad.astype(np.float32), SQUARE)
        lum = board_lum(img, quad)
        grip = gripper_mask(img)
        if i % lib_stride == 0:
            for cls, box in anns:
                x, y, bw, bh = (int(v) for v in box)
                if grip[max(0, y):y + bh, max(0, x):x + bw].mean() > 0.01:
                    continue  # 被爪子挡到的螃蟹不当素材
                got = lift(img, H, box)
                if got is not None:
                    lib[cls].append((*got, lum))
        prepared.append((img, anns, H, lum, quad, grip))
    print(f"  可用帧 {len(prepared)}，螃蟹素材 " + ", ".join(f"{CLASSES[c]} {len(v)}" for c, v in lib.items()))
    out = []
    for img, anns, H, lum, quad, grip in prepared:
        h, w = img.shape[:2]
        board = np.zeros((h, w), np.uint8)
        cv2.fillPoly(board, [quad.astype(np.int32)], 1)
        board = cv2.dilate(board, np.ones((25, 25), np.uint8))
        # 挨着爪子或板子边缘的螃蟹抹不干净：原样保留（标签也保留），其余的抹掉
        kept, erased = [], []
        keepmask = np.zeros((h, w), bool)
        for cls, box in anns:
            x, y, bw, bh = box
            px, py = 0.14 * bw + 3, 0.14 * bh + 3
            x0, y0, x1, y1 = int(x - px), int(y - py), int(x + bw + px), int(y + bh + py)
            if x0 < 0 or y0 < 0 or x1 > w or y1 > h or grip[y0:y1, x0:x1].any() or board[y0:y1, x0:x1].mean() < 0.9:
                kept.append((cls, float(x), float(y), float(bw), float(bh)))
                keepmask[max(0, y0):y1, max(0, x0):x1] = True
            else:
                erased.append(box)
        bg = erase(img, erased, quad, grip)
        Hinv = np.linalg.inv(H)
        for _ in range(per_frame):
            canvas = bg.copy()
            placed, new = [], list(kept)
            have = [sum(1 for k in kept if k[0] == c) for c in range(3)]
            want = [c for c, n in enumerate((rng.integers(0, 7), rng.integers(0, 6), rng.integers(0, 6)))
                    for _ in range(max(0, int(n) - have[c]))]
            rng.shuffle(want)
            for cls in want:
                for _try in range(25):
                    patch, alpha, r, k, plum = lib[cls][int(rng.integers(len(lib[cls])))]
                    c = rng.uniform(r * 0.8 + 10, N - r * 0.8 - 10, 2)
                    if any(np.linalg.norm(c - pc) < 0.72 * (r + pr) for pc, pr in placed):
                        continue
                    S = patch.shape[0]
                    th = rng.uniform(0, 2 * math.pi)
                    cs, sn = math.cos(th) / k, math.sin(th) / k
                    T = np.array([[cs, -sn, c[0] - (cs - sn) * S / 2], [sn, cs, c[1] - (sn + cs) * S / 2], [0, 0, 1]])
                    M = Hinv @ T
                    a = cv2.warpPerspective(alpha, M, (w, h), flags=cv2.INTER_LINEAR)
                    core = a > 0.5
                    area = int(core.sum())
                    if area < 150 or area > 1.6 * float((alpha > 0.5).sum()):
                        continue  # 太小，或者要放大太多（会糊）
                    ys, xs = np.nonzero(core)
                    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
                    if x0 <= 1 or y0 <= 1 or x1 >= w - 2 or y1 >= h - 2:
                        continue
                    if grip[core].mean() > 0.02 or keepmask[a > 0.05].mean() > 0.02:
                        continue  # 不贴到爪子或保留的螃蟹上
                    warped = cv2.warpPerspective(patch, M, (w, h), flags=cv2.INTER_LINEAR)
                    gain = float(np.clip(lum / max(plum, 1), 0.75, 1.35))
                    a3 = a[..., None]
                    canvas = canvas * (1 - a3) + warped.astype(np.float32) * gain * a3
                    placed.append((c, r))
                    new.append((cls, float(x0), float(y0), float(x1 - x0 + 1), float(y1 - y0 + 1)))
                    break
            res = np.clip(canvas, 0, 255).astype(np.uint8)
            if rng.random() < 0.5:  # 左右翻转
                res = res[:, ::-1].copy()
                new = [(cl, w - x - bw, y, bw, bh) for cl, x, y, bw, bh in new]
            out.append((res, new))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--per-frame", type=int, default=5)
    ap.add_argument("--preview", type=int, default=0)
    a = ap.parse_args()
    base = ROOT / "real" / a.name
    log = json.loads((base / "_log.json").read_text())
    clean = {f"{a.name}_{int(k[2:6]):04d}.jpg" for k, v in log.items() if v["projected"] == 0 and v["occluded"] == 0}
    out = ROOT / "real" / f"{a.name}_cp"
    cats = [{"id": i + 1, "name": c, "supercategory": "crab"} for i, c in enumerate(CLASSES)]
    for split, seed in (("train", 11), ("valid", 12)):
        frames = load_split(base, split, clean)
        rng = np.random.default_rng(seed)
        if a.preview:
            frames = frames[:: max(1, len(frames) // a.preview)][: a.preview]
        print(split)
        gen = build(frames, rng, 1 if a.preview else (a.per_frame if split == "train" else 2), 1 if a.preview else 2)
        d = out / ("_preview" if a.preview else split)
        d.mkdir(parents=True, exist_ok=True)
        coco = {"images": [], "annotations": [], "categories": cats}
        for i, (img, anns) in enumerate(gen):
            name = f"{a.name}_cp_{split}_{i:05d}.jpg"
            if a.preview:
                for cls, x, y, w, h in anns:
                    cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), COLORS[cls], 2)
            cv2.imwrite(str(d / name), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            coco["images"].append({"id": i, "file_name": name, "width": img.shape[1], "height": img.shape[0]})
            for cls, x, y, w, h in anns:
                coco["annotations"].append({"id": len(coco["annotations"]), "image_id": i, "category_id": cls + 1,
                                            "bbox": [round(v, 1) for v in (x, y, w, h)], "area": round(w * h, 1), "iscrowd": 0})
        if not a.preview:
            (d / "_annotations.coco.json").write_text(json.dumps(coco))
        print(f"  {split}: {len(gen)} 张图, {len(coco['annotations'])} 个框")
        if a.preview:
            break


if __name__ == "__main__":
    main()
