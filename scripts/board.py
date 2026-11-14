"""在整幅 ROV 画面里找到白色样本板，返回它的外接框（带一点边距）。

ROV 是广角镜头，板子只占画面一小块；直接把 1920x1080 缩到 512 送进模型，
螃蟹只剩十几个像素。所以先找到板子、裁出来，再识别。
"""
from __future__ import annotations

import cv2
import numpy as np


def find_board(bgr: np.ndarray, margin: float = 0.06):
    """返回 (x0, y0, x1, y1)；找不到返回 None。"""
    h, w = bgr.shape[:2]
    s = 480 / w
    small = cv2.resize(bgr, (480, int(h * s)), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    # 板子：低饱和度 + 亮；池底瓷砖是高饱和度的蓝
    mask = ((hsv[..., 1] < 70) & (hsv[..., 2] > 150)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best, best_score = None, 0.0
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 0.01 * mask.size:
            continue
        hull = cv2.convexHull(c)
        solidity = area / max(cv2.contourArea(hull), 1)
        x, y, bw, bh = cv2.boundingRect(hull)
        # 排除画面顶部的水面反光/墙：它们又扁又贴着上边缘
        if y <= 1 and bh < 0.25 * mask.shape[0]:
            continue
        score = area * solidity
        if score > best_score:
            best, best_score = (x, y, bw, bh), score
    if best is None:
        return None
    x, y, bw, bh = best
    m = margin * max(bw, bh)
    x0 = int(max(0, (x - m) / s)); y0 = int(max(0, (y - m) / s))
    x1 = int(min(w, (x + bw + m) / s)); y1 = int(min(h, (y + bh + m) / s))
    return x0, y0, x1, y1


def board_mask(bgr: np.ndarray, width: int = 640):
    h, w = bgr.shape[:2]
    s = width / w
    small = cv2.resize(bgr, (width, int(h * s)), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    mask = ((hsv[..., 1] < 70) & (hsv[..., 2] > 150)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    return mask, s


def find_quad(bgr: np.ndarray):
    """板子的四个角（原图坐标，顺时针，从最靠左上的角开始）；找不到返回 None。"""
    mask, s = board_mask(bgr)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best, best_score = None, 0.0
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 0.01 * mask.size:
            continue
        hull = cv2.convexHull(c)
        x, y, bw, bh = cv2.boundingRect(hull)
        if y <= 1 and bh < 0.25 * mask.shape[0]:
            continue
        score = area * area / max(cv2.contourArea(hull), 1)
        if score > best_score:
            best, best_score = hull, score
    if best is None:
        return None
    peri = cv2.arcLength(best, True)
    quad = None
    for k in np.linspace(0.01, 0.12, 23):
        ap = cv2.approxPolyDP(best, k * peri, True)
        if len(ap) == 4:
            quad = ap[:, 0, :].astype(np.float32)
            break
    if quad is None:
        return None
    # 顺时针排序
    c = quad.mean(0)
    ang = np.arctan2(quad[:, 1] - c[1], quad[:, 0] - c[0])
    quad = quad[np.argsort(ang)]
    i0 = int(np.argmin(quad[:, 0] + quad[:, 1]))
    quad = np.roll(quad, -i0, axis=0)
    return quad / s
