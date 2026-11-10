"""合成「入侵蟹样本板」训练图。

照 2026 NAVIGATOR Task 2.1 的道具来做（见 rules/NV_Prop_2026.pdf 第 4 页实拍）：
  50x50 cm 白色瓦楞塑料板，上面贴过塑、剪成小片的螃蟹打印图（连着原图背景），
  透明胶带固定；图片会缩放、任意角度旋转；只用手册里那三张图。

每张合成图 = 背景（蓝色防水布 / 泳池瓷砖 / 水体 / 水泥地）
          + 透视变换后的样本板
          + 光照、反光、水下偏色、模糊、噪声、JPEG 等退化。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CRAB_DIR = ROOT / "assets/crabs"

CLASSES = ["green", "rock", "jonah"]  # COCO category id = index + 1
BOARD = 1000  # 样本板在「板坐标」里的边长（像素），对应 50 cm


@dataclass
class Crab:
    rgb: np.ndarray   # HxWx3 uint8，打印出来的整张图
    mask: np.ndarray  # HxW uint8，螃蟹本体


def load_crabs() -> list[Crab]:
    out = []
    for name in CLASSES:
        rgba = cv2.imread(str(CRAB_DIR / f"{name}.png"), cv2.IMREAD_UNCHANGED)
        out.append(Crab(rgba[:, :, :3].copy(), rgba[:, :, 3].copy()))
    return out


# ---------------------------------------------------------------- 工具函数

def smooth_noise(rng: np.random.Generator, h: int, w: int, scale: int) -> np.ndarray:
    """0~1 的低频噪声场。"""
    sh, sw = max(2, h // scale), max(2, w // scale)
    n = rng.random((sh, sw)).astype(np.float32)
    # 在小图上模糊再放大，比在全分辨率上做大半径模糊快得多
    n = cv2.GaussianBlur(n, (0, 0), 0.6, borderType=cv2.BORDER_REFLECT)
    n = cv2.resize(n, (w, h), interpolation=cv2.INTER_CUBIC)
    n -= n.min()
    return n / max(n.max(), 1e-6)


def alpha_blend(dst: np.ndarray, src: np.ndarray, a: np.ndarray) -> None:
    """dst = dst*(1-a) + src*a，原地；a 为 HxW float32 0~1。"""
    a3 = a[..., None]
    dst[:] = dst * (1 - a3) + src * a3


# ---------------------------------------------------------------- 背景

def make_background(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    kind = rng.choice(["tarp", "tile", "water", "concrete", "dark"], p=[0.3, 0.25, 0.25, 0.1, 0.1])
    img = np.zeros((h, w, 3), np.float32)
    if kind == "tarp":
        base = np.array([rng.uniform(140, 210), rng.uniform(60, 110), rng.uniform(10, 50)], np.float32)  # BGR 蓝
        wrinkle = smooth_noise(rng, h, w, int(rng.integers(20, 80)))
        img[:] = base * (0.6 + 0.6 * wrinkle[..., None])
    elif kind == "tile":
        base = np.array([rng.uniform(170, 240), rng.uniform(150, 220), rng.uniform(90, 170)], np.float32)
        img[:] = base
        step = int(rng.integers(25, 90))
        grout = base * rng.uniform(0.6, 0.85)
        off = int(rng.integers(0, step))
        for x in range(off, w, step):
            img[:, x:x + 2] = grout
        for y in range(off, h, step):
            img[y:y + 2, :] = grout
        img *= (0.75 + 0.35 * smooth_noise(rng, h, w, 60))[..., None]
    elif kind == "water":
        top = np.array([rng.uniform(120, 220), rng.uniform(110, 200), rng.uniform(20, 90)], np.float32)
        bot = top * rng.uniform(0.3, 0.7)
        t = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
        img[:] = top * (1 - t) + bot * t
    elif kind == "concrete":
        g = rng.uniform(100, 180)
        img[:] = g
        img += (rng.normal(0, 12, (h, w, 1))).astype(np.float32)
        img *= (0.8 + 0.3 * smooth_noise(rng, h, w, 40))[..., None]
    else:
        img[:] = np.array([rng.uniform(30, 80), rng.uniform(30, 70), rng.uniform(5, 30)], np.float32)
    return np.clip(img, 0, 255)


# ---------------------------------------------------------------- 板子

def make_board(rng: np.random.Generator) -> np.ndarray:
    v = rng.uniform(225, 250)
    board = np.full((BOARD, BOARD, 3), v, np.float32)
    board += np.array([rng.uniform(-6, 6), rng.uniform(-4, 4), rng.uniform(-6, 6)], np.float32)
    # 瓦楞纹路
    period = rng.uniform(6, 12)
    amp = rng.uniform(1.0, 5.0)
    xs = np.arange(BOARD, dtype=np.float32)
    flute = amp * np.sin(2 * np.pi * xs / period)
    if rng.random() < 0.5:
        board += flute[None, :, None]
    else:
        board += flute[:, None, None]
    # 污渍、轻微不均
    board *= (0.92 + 0.1 * smooth_noise(rng, BOARD, BOARD, 120))[..., None]
    # 四角的扣眼
    for cx, cy in [(45, 45), (955, 45), (45, 955), (955, 955)]:
        cx += int(rng.integers(-15, 15)); cy += int(rng.integers(-15, 15))
        r = int(rng.integers(8, 14))
        cv2.circle(board, (cx, cy), r + 4, (150, 150, 150), -1, cv2.LINE_AA)
        cv2.circle(board, (cx, cy), r, (40, 40, 40), -1, cv2.LINE_AA)
    return board


def patch_shape(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    """剪下来的那一片的形状（手剪的椭圆 / 圆角矩形）。"""
    m = np.zeros((h, w), np.uint8)
    if rng.random() < 0.5:
        ax = (int(w / 2 * rng.uniform(1.0, 1.18)), int(h / 2 * rng.uniform(1.0, 1.25)))
        pts = cv2.ellipse2Poly((w // 2, h // 2), ax, 0, 0, 360, 8)
    else:
        r = int(min(h, w) * rng.uniform(0.05, 0.35))
        rect = np.zeros((h, w), np.uint8)
        cv2.rectangle(rect, (r, 0), (w - r, h), 255, -1)
        cv2.rectangle(rect, (0, r), (w, h - r), 255, -1)
        for cx, cy in [(r, r), (w - r, r), (r, h - r), (w - r, h - r)]:
            cv2.circle(rect, (cx, cy), r, 255, -1)
        cnts, _ = cv2.findContours(rect, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        pts = cnts[0][:, 0, :][::6]
    # 手剪的抖动
    pts = pts.astype(np.float32) + rng.normal(0, min(h, w) * 0.012, pts.shape).astype(np.float32)
    cv2.fillPoly(m, [pts.astype(np.int32)], 255)
    return m


def make_patch(rng: np.random.Generator, crab: Crab, size: float, angle: float):
    """返回 (rgb, alpha, crabmask)，都已缩放+旋转，float32/uint8。"""
    h, w = crab.mask.shape
    # 在原图周围随机加一圈白纸边（打印纸）
    pad = int(rng.uniform(0, 0.12) * max(h, w))
    paper = rng.uniform(235, 255)
    rgb = cv2.copyMakeBorder(crab.rgb, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(paper,) * 3)
    cm = cv2.copyMakeBorder(crab.mask, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    H, W = cm.shape
    shape = patch_shape(rng, H, W)
    cm = cv2.bitwise_and(cm, shape)  # 被剪掉的腿不算
    # 过塑膜：比纸片大一圈的透明边
    lam_px = int(rng.uniform(0.01, 0.05) * max(H, W)) + 1
    lam = cv2.dilate(shape, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * lam_px + 1,) * 2))
    alpha = np.where(shape > 0, 1.0, np.where(lam > 0, rng.uniform(0.12, 0.3), 0.0)).astype(np.float32)
    lam_color = np.full_like(rgb, rng.uniform(225, 250))
    rgb = np.where(shape[..., None] > 0, rgb, lam_color).astype(np.float32)
    # 留出旋转空间
    m = int(lam_px + 2)
    D = int(math.ceil(math.hypot(H, W))) + 2 * m
    oy, ox = (D - H) // 2, (D - W) // 2
    big_rgb = np.zeros((D, D, 3), np.float32); big_rgb[oy:oy + H, ox:ox + W] = rgb
    big_a = np.zeros((D, D), np.float32); big_a[oy:oy + H, ox:ox + W] = alpha
    big_m = np.zeros((D, D), np.uint8); big_m[oy:oy + H, ox:ox + W] = cm
    k = size / max(h, w)
    out = max(8, int(D * k))
    M = cv2.getRotationMatrix2D((D / 2, D / 2), angle, k)
    M[:, 2] += out / 2 - D / 2
    big_rgb = cv2.warpAffine(big_rgb, M, (out, out), flags=cv2.INTER_AREA if k < 1 else cv2.INTER_LINEAR)
    big_a = cv2.warpAffine(big_a, M, (out, out), flags=cv2.INTER_LINEAR)
    big_m = cv2.warpAffine(big_m, M, (out, out), flags=cv2.INTER_NEAREST)
    # 裁到非透明区域
    ys, xs = np.nonzero(big_a > 0.01)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    return big_rgb[y0:y1, x0:x1], big_a[y0:y1, x0:x1], big_m[y0:y1, x0:x1]


def add_tape(rng: np.random.Generator, board: np.ndarray, x: int, y: int, ph: int, pw: int) -> None:
    """在纸片边缘贴 1~3 条透明胶带。"""
    for _ in range(int(rng.integers(1, 4))):
        tw = rng.uniform(30, 50)  # 胶带宽 ~2 cm
        tl = rng.uniform(0.5, 1.2) * max(ph, pw)
        cx = x + rng.uniform(0, pw); cy = y + rng.uniform(0, ph)
        ang = rng.uniform(0, 180)
        box = cv2.boxPoints(((cx, cy), (tl, tw), ang)).astype(np.int32)
        # 只在胶带外接框这一小块里混合，不碰整块板
        x0, y0 = np.clip(box.min(0) - 2, 0, BOARD); x1, y1 = np.clip(box.max(0) + 3, 0, BOARD)
        if x1 <= x0 or y1 <= y0:
            continue
        a = np.zeros((y1 - y0, x1 - x0), np.float32)
        local = box - np.array([x0, y0], np.int32)
        cv2.fillPoly(a, [local], float(rng.uniform(0.08, 0.25)))
        if rng.random() < 0.5:  # 胶带边缘的反光线
            cv2.polylines(a, [local], True, float(rng.uniform(0.15, 0.35)), 1, cv2.LINE_AA)
        region = board[y0:y1, x0:x1]
        alpha_blend(region, np.float32(rng.uniform(200, 255)), a)


def populate_board(rng: np.random.Generator, crabs: list[Crab], board: np.ndarray):
    """往板上贴螃蟹，返回 [(class_idx, crab_mask_in_board_coords_points)]。"""
    if rng.random() < 0.04:
        return []
    counts = [int(rng.integers(0, 7)), int(rng.integers(0, 6)), int(rng.integers(0, 6))]
    order = [c for c, n in enumerate(counts) for _ in range(n)]
    rng.shuffle(order)
    placed_boxes: list[tuple[int, int, int, int]] = []
    items = []
    for cls in order:
        placed = False
        # 纸片最多做 3 次（每次越做越小）；每张纸片只换位置，不重新生成
        for shrink in (1.0, 0.75, 0.55):
            size = rng.uniform(110, 360) * shrink
            rgb, a, cm = make_patch(rng, crabs[cls], size, rng.uniform(0, 360))
            ph, pw = a.shape
            if ph >= BOARD - 60 or pw >= BOARD - 60:
                continue
            for _ in range(30):
                x = int(rng.integers(20, BOARD - pw - 20)); y = int(rng.integers(20, BOARD - ph - 20))
                # 纸片之间允许少量重叠，但不能盖住别的螃蟹
                box = (x, y, x + pw, y + ph)
                if any(_overlap(box, b) > 0.08 for b in placed_boxes):
                    continue
                ys, xs = np.nonzero(cm)
                if len(xs) < 30:
                    break
                alpha_blend(board[y:y + ph, x:x + pw], rgb, a)
                add_tape(rng, board, x, y, ph, pw)
                items.append((cls, np.stack([xs + x, ys + y], 1).astype(np.float32)))
                placed_boxes.append(box)
                placed = True
                break
            if placed:
                break
    return items


def _overlap(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter == 0:
        return 0.0
    small = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return inter / small


# ---------------------------------------------------------------- 退化

def degrade(rng: np.random.Generator, img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    underwater = rng.random() < 0.75
    # 不均匀光照 + 暗角
    light = 0.55 + 0.7 * smooth_noise(rng, h, w, int(rng.integers(150, 500)))
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.hypot((xx - w / 2) / w, (yy - h / 2) / h)
    light *= 1 - rng.uniform(0, 0.8) * r ** 2
    img *= light[..., None]
    # 过塑膜/水面反光
    for _ in range(int(rng.integers(0, 3 if underwater else 4))):
        g = np.zeros((h, w), np.float32)
        cv2.ellipse(g, (int(rng.uniform(0, w)), int(rng.uniform(0, h))),
                    (int(rng.uniform(10, w / 5)), int(rng.uniform(5, h / 8))), rng.uniform(0, 180), 0, 360, 1.0, -1)
        g = cv2.GaussianBlur(g, (0, 0), rng.uniform(4, 25))
        img += g[..., None] * rng.uniform(60, 220)
    if underwater:
        # 红光衰减 + 蓝绿偏色 + 雾
        hard = rng.random() < 0.2  # 少量很浑的样本
        img[..., 2] *= rng.uniform(0.45, 0.8) if hard else rng.uniform(0.7, 0.97)
        img[..., 1] *= rng.uniform(0.85, 1.0)
        water = np.array([rng.uniform(110, 210), rng.uniform(100, 190), rng.uniform(10, 80)], np.float32)
        a = rng.uniform(0.25, 0.45) if hard else rng.uniform(0.03, 0.25)
        img = img * (1 - a) + water * a
        # 焦散纹
        if rng.random() < 0.5:
            c = smooth_noise(rng, h, w, int(rng.integers(15, 40)))
            c = np.abs(np.sin(c * rng.uniform(8, 16)))
            img *= (1 + rng.uniform(0.03, 0.12) * (c - 0.5))[..., None]
        # 水中悬浮颗粒
        for _ in range(int(rng.integers(0, 60))):
            cv2.circle(img, (int(rng.uniform(0, w)), int(rng.uniform(0, h))), int(rng.integers(1, 3)),
                       (rng.uniform(180, 255),) * 3, -1, cv2.LINE_AA)
    # 色彩微调（色相只能小幅度动，螃蟹靠颜色区分）
    img = np.clip(img, 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + rng.uniform(-5, 5)) % 180
    hsv[..., 1] *= rng.uniform(0.7, 1.2)
    img = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
    img = img * rng.uniform(0.75, 1.25) + rng.uniform(-25, 25)
    gamma = rng.uniform(0.75, 1.35)
    img = 255 * (np.clip(img, 0, 255) / 255) ** gamma
    # 模糊
    p = rng.random()
    if p < 0.35:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.5, 2.2))
    elif p < 0.55:
        k = int(rng.integers(3, 12))
        ker = np.zeros((k, k), np.float32); ker[k // 2, :] = 1 / k
        ker = cv2.warpAffine(ker, cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), rng.uniform(0, 180), 1), (k, k))
        img = cv2.filter2D(img, -1, ker / max(ker.sum(), 1e-6))
    # 低分辨率摄像头
    if rng.random() < 0.5:
        f = rng.uniform(0.35, 0.9)
        small = cv2.resize(img, (max(16, int(w * f)), max(16, int(h * f))), interpolation=cv2.INTER_AREA)
        img = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    img += rng.normal(0, rng.uniform(0, 9), img.shape).astype(np.float32)
    img = np.clip(img, 0, 255).astype(np.uint8)
    q = int(rng.integers(35, 96))
    return cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])[1], cv2.IMREAD_COLOR)


# ---------------------------------------------------------------- 组合

SIZES = [(640, 480), (800, 600), (960, 540), (1280, 720), (720, 720)]


def render(rng: np.random.Generator, crabs: list[Crab]):
    """返回 (BGR uint8 图, [(class_idx, x, y, w, h)])。"""
    while True:
        w, h = SIZES[int(rng.integers(len(SIZES)))]
        board = make_board(rng)
        items = populate_board(rng, crabs, board)
        # 板子在画面里：大小、位置、旋转、透视
        L = rng.uniform(0.55, 1.5) * min(w, h)
        cx = w / 2 + rng.uniform(-0.3, 0.3) * w
        cy = h / 2 + rng.uniform(-0.3, 0.3) * h
        th = math.radians(rng.uniform(-30, 30))
        sq = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], np.float32) * L / 2
        R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]], np.float32)
        quad = sq @ R.T + np.array([cx, cy], np.float32)
        quad += rng.normal(0, 0.06 * L, quad.shape).astype(np.float32)
        src = np.array([[0, 0], [BOARD, 0], [BOARD, BOARD], [0, BOARD]], np.float32)
        Hm = cv2.getPerspectiveTransform(src, quad)

        anns = []
        ok = True
        for cls, pts in items:
            p = cv2.perspectiveTransform(pts[None], Hm)[0]
            inside = (p[:, 0] >= 0) & (p[:, 0] < w) & (p[:, 1] >= 0) & (p[:, 1] < h)
            frac = inside.mean()
            if frac < 0.02:
                continue  # 完全在画面外
            if frac < 0.6:
                ok = False  # 只露出一半的螃蟹，标不标都不对，换一张
                break
            q = p[inside]
            x0, y0 = q.min(0); x1, y1 = q.max(0)
            if (x1 - x0) < 10 or (y1 - y0) < 10:
                continue
            anns.append((cls, float(x0), float(y0), float(x1 - x0 + 1), float(y1 - y0 + 1)))
        if not ok:
            continue

        img = make_background(rng, h, w)
        warped = cv2.warpPerspective(board, Hm, (w, h), flags=cv2.INTER_AREA)
        a = cv2.warpPerspective(np.ones((BOARD, BOARD), np.float32), Hm, (w, h), flags=cv2.INTER_LINEAR)
        # 板子投在背景上的阴影
        if rng.random() < 0.6:
            sh = cv2.GaussianBlur(np.roll(a, (int(rng.integers(3, 15)), int(rng.integers(3, 15))), (0, 1)), (0, 0), 6)
            img *= (1 - 0.4 * sh)[..., None]
        alpha_blend(img, warped, a)
        return degrade(rng, img), anns
