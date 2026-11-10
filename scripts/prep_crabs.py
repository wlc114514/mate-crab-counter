"""从手册第 27 页的官方图切出三只螃蟹，并用 GrabCut 算出螃蟹本体的掩膜。

输出 assets/crabs/<name>.png（RGBA，alpha = 螃蟹掩膜），合成时：
  - RGB 整张当作「打印出来的图片」（含原来的背景）
  - alpha 只用来算螃蟹本体的标注框
"""
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "assets/source/manual_p27_crabs.jpg"
OUT = ROOT / "assets/crabs"

# 列范围由白色分隔列确定（464-475、856-863 是纯白分隔）
CROPS = {
    "green": (0, 464),   # 欧洲青蟹（入侵种，要数的）
    "rock": (476, 856),  # 本地 Rock crab
    "jonah": (864, 1216),  # 本地 Jonah crab
}


def crab_mask(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    rect = (4, 4, w - 8, h - 8)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.grabCut(img, mask, rect, bgd, fgd, 8, cv2.GC_INIT_WITH_RECT)
    m = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    # 去小噪点，只保留最大连通域及其附近的腿
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    keep = np.zeros_like(m)
    if n > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        big = areas.max()
        for i, a in enumerate(areas, start=1):
            if a >= 0.01 * big:
                keep[lab == i] = 255
    return keep


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    src = cv2.imread(str(SRC))
    preview = []
    for name, (x0, x1) in CROPS.items():
        crop = src[:, x0:x1].copy()
        m = crab_mask(crop)
        ys, xs = np.nonzero(m)
        print(f"{name}: crop {crop.shape[1]}x{crop.shape[0]}, crab bbox x{xs.min()}-{xs.max()} y{ys.min()}-{ys.max()}")
        rgba = np.dstack([crop, m])
        cv2.imwrite(str(OUT / f"{name}.png"), rgba)
        vis = crop.copy()
        vis[m == 0] = (vis[m == 0] * 0.3).astype(np.uint8)
        cv2.rectangle(vis, (int(xs.min()), int(ys.min())), (int(xs.max()), int(ys.max())), (0, 0, 255), 2)
        preview.append(vis)
    h = max(p.shape[0] for p in preview)
    cv2.imwrite(str(OUT / "_mask_preview.jpg"), np.hstack([cv2.copyMakeBorder(p, 0, h - p.shape[0], 0, 6, cv2.BORDER_CONSTANT, value=(0, 255, 0)) for p in preview]))


if __name__ == "__main__":
    main()
