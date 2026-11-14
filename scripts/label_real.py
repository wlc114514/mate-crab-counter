"""给真实视频帧自动打标签（同一块板、摆法不变的一段视频）。

思路：螃蟹在板上的位置是固定的。先在一帧拉正的俯视图上人工确认每只的种类（基准布局），
之后每一帧：找到板子四角 -> 把检测框中心映射到板坐标 -> 按位置对应到基准布局拿到正确种类。
框的位置用模型给的（定位是准的，错的只是种类）；没检测到但没被爪子挡住的螃蟹，用基准框投影回去补。

输出 COCO 数据集（图片是「裁出板子」后的画面）：
  real/<name>/{train,valid}/_annotations.coco.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parent))
from board import find_quad  # noqa: E402
from count import ROOT, load  # noqa: E402

N = 800  # 基准俯视图边长
CLASSES = ["green", "rock", "jonah"]
COLORS = [(0, 220, 0), (0, 140, 255), (255, 0, 255)]
SQUARE = np.float32([[0, 0], [N, 0], [N, N], [0, N]])


def load_canon(path: Path, classes: dict[int, str]):
    d = json.loads(path.read_text())
    boxes = np.array([d["dets"][i][:4] for i in classes], np.float32)
    cls = [CLASSES.index({"g": "green", "r": "rock", "j": "jonah"}[c]) for c in classes.values()]
    return boxes, cls


def match(centers: np.ndarray, canon_c: np.ndarray, H: np.ndarray, max_d: float):
    p = cv2.perspectiveTransform(centers[None].astype(np.float32), H)[0]
    cost = np.linalg.norm(p[:, None] - canon_c[None], axis=2)
    r, c = linear_sum_assignment(cost)
    keep = cost[r, c] < max_d
    return r[keep], c[keep], float(cost[r, c][keep].sum())


def label_frame(bgr, model, canon_boxes, canon_cls, thr=0.25):
    """返回 (crop_box, [(cls, x, y, w, h)]（裁剪图坐标）, info) 或 None。"""
    quad = find_quad(bgr)
    if quad is None:
        return None
    h, w = bgr.shape[:2]
    x0, y0 = quad.min(0); x1, y1 = quad.max(0)
    m = 0.08 * max(x1 - x0, y1 - y0)
    cx0, cy0 = int(max(0, x0 - m)), int(max(0, y0 - m))
    cx1, cy1 = int(min(w, x1 + m)), int(min(h, y1 + m))
    crop = bgr[cy0:cy1, cx0:cx1]
    det = model.predict(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)), threshold=thr)
    if len(det) < 6:
        return None
    boxes = det.xyxy.astype(np.float32) + np.float32([cx0, cy0, cx0, cy0])  # 整幅画面坐标
    centers = np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2], 1)
    canon_c = np.stack([(canon_boxes[:, 0] + canon_boxes[:, 2]) / 2, (canon_boxes[:, 1] + canon_boxes[:, 3]) / 2], 1)

    best = None
    for rot in range(4):
        H = cv2.getPerspectiveTransform(np.roll(quad, -rot, axis=0).astype(np.float32), SQUARE)
        r, c, cost = match(centers, canon_c, H, 110)
        key = (len(r), -cost)
        if best is None or key > best[0]:
            best = (key, H, r, c)
    _, H, r, c = best
    if len(r) < 8:
        return None
    # 用匹配上的点重新估计单应矩阵，再匹配一遍
    for _ in range(2):
        H2, _ = cv2.findHomography(centers[r], canon_c[c], cv2.RANSAC, 25)
        if H2 is None:
            break
        r2, c2, _ = match(centers, canon_c, H2, 60)
        if len(r2) >= len(r) - 1:
            H, r, c = H2, r2, c2
    resid = np.linalg.norm(cv2.perspectiveTransform(centers[r][None], H)[0] - canon_c[c], axis=1)

    anns = []
    for i, j in zip(r, c):
        bx = boxes[i]
        anns.append((canon_cls[j], bx[0] - cx0, bx[1] - cy0, bx[2] - bx[0], bx[3] - bx[1]))
    # 没匹配到的基准螃蟹：没被爪子挡住就用投影框补上
    Hinv = np.linalg.inv(H)
    hsv_v = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[..., 2]
    occluded, projected = 0, 0
    for j in set(range(len(canon_cls))) - set(c.tolist()):
        b = canon_boxes[j]
        corners = np.float32([[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]])
        p = cv2.perspectiveTransform(corners[None], Hinv)[0]
        px0, py0 = p.min(0); px1, py1 = p.max(0)
        if px0 < cx0 or py0 < cy0 or px1 > cx1 or py1 > cy1:
            occluded += 1
            continue
        patch = hsv_v[int(py0):int(py1), int(px0):int(px1)]
        if patch.size == 0 or (patch < 55).mean() > 0.35:
            occluded += 1  # 大片黑色 = 被爪子挡住
            continue
        projected += 1
        anns.append((canon_cls[j], px0 - cx0, py0 - cy0, px1 - px0, py1 - py0))
    info = {"matched": int(len(r)), "projected": projected, "occluded": occluded,
            "resid": float(resid.mean()), "ndet": int(len(det))}
    return (cx0, cy0, cx1, cy1), anns, info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("frames")
    ap.add_argument("--canon", required=True)
    ap.add_argument("--classes", required=True, help='如 "0:j,1:j,3:r"（基准检测序号:种类）')
    ap.add_argument("--name", required=True)
    ap.add_argument("--valid-from", type=int, required=True, help="帧序号 >= 这个值的进 valid")
    ap.add_argument("--max-resid", type=float, default=30)
    a = ap.parse_args()
    classes = {int(k): v for k, v in (kv.split(":") for kv in a.classes.split(","))}
    canon_boxes, canon_cls = load_canon(Path(a.canon), classes)
    model = load(str(ROOT / "runs/small/checkpoint_best_total.pth"))
    out = ROOT / "real" / a.name
    splits = {s: {"images": [], "annotations": []} for s in ("train", "valid")}
    for s in splits:
        (out / s).mkdir(parents=True, exist_ok=True)
    (out / "_vis").mkdir(exist_ok=True)
    stats = {"used": 0, "no_board": 0, "bad_fit": 0, "projected": 0, "occluded": 0}
    log = {}
    for f in sorted(Path(a.frames).glob("f_*.jpg")):
        idx = int(f.stem.split("_")[1])
        bgr = cv2.imread(str(f))
        res = label_frame(bgr, model, canon_boxes, canon_cls)
        if res is None:
            stats["no_board"] += 1
            continue
        (cx0, cy0, cx1, cy1), anns, info = res
        log[f.name] = info
        if info["resid"] > a.max_resid:
            stats["bad_fit"] += 1
            continue
        split = "valid" if idx >= a.valid_from else "train"
        d = splits[split]
        crop = bgr[cy0:cy1, cx0:cx1]
        name = f"{a.name}_{idx:04d}.jpg"
        cv2.imwrite(str(out / split / name), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        iid = len(d["images"])
        d["images"].append({"id": iid, "file_name": name, "width": crop.shape[1], "height": crop.shape[0]})
        vis = crop.copy()
        for cls, x, y, w, h in anns:
            d["annotations"].append({"id": len(d["annotations"]), "image_id": iid, "category_id": cls + 1,
                                     "bbox": [round(float(v), 1) for v in (x, y, w, h)],
                                     "area": round(float(w * h), 1), "iscrowd": 0})
            cv2.rectangle(vis, (int(x), int(y)), (int(x + w), int(y + h)), COLORS[cls], 2)
        cv2.putText(vis, f"{idx} m{info['matched']} p{info['projected']} o{info['occluded']}", (6, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imwrite(str(out / "_vis" / name), vis)
        stats["used"] += 1; stats["projected"] += info["projected"]; stats["occluded"] += info["occluded"]
    cats = [{"id": i + 1, "name": c, "supercategory": "crab"} for i, c in enumerate(CLASSES)]
    for s, d in splits.items():
        (out / s / "_annotations.coco.json").write_text(json.dumps({**d, "categories": cats}))
        print(s, len(d["images"]), "张图,", len(d["annotations"]), "个框")
    (out / "_log.json").write_text(json.dumps(log))
    print(stats)


if __name__ == "__main__":
    main()
