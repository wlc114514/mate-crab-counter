"""组装微调数据集：真实帧（只用 14 只全部对上的干净帧）+ 一部分合成数据。

用法： python scripts/make_ft_dataset.py --real pool1112:1 pool1112_cp:1 --synth 1200   （名字:训练时重复次数）
输出： dataset_ft/{train,valid,test}
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "dataset_ft"


class Coco:
    def __init__(self, cats):
        self.d = {"images": [], "annotations": [], "categories": cats}

    def add(self, src_dir: Path, img: dict, anns: list[dict], dst_dir: Path, new_name: str | None = None):
        name = new_name or img["file_name"]
        shutil.copyfile(src_dir / img["file_name"], dst_dir / name)
        iid = len(self.d["images"])
        self.d["images"].append({**img, "id": iid, "file_name": name})
        for a in anns:
            self.d["annotations"].append({**a, "id": len(self.d["annotations"]), "image_id": iid})


def by_image(coco: dict) -> dict[int, list[dict]]:
    m: dict[int, list[dict]] = {}
    for a in coco["annotations"]:
        m.setdefault(a["image_id"], []).append(a)
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", nargs="+", required=True)
    ap.add_argument("--test", default=None, help="test 用哪个真实数据集的 valid（默认第一个）")
    ap.add_argument("--synth", type=int, default=1200)
    ap.add_argument("--synth-valid", type=int, default=100)
    ap.add_argument("--out", default="dataset_ft")
    a = ap.parse_args()
    global OUT
    OUT = ROOT / a.out
    random.seed(0)
    if OUT.exists():
        shutil.rmtree(OUT)
    for s in ("train", "valid", "test"):
        (OUT / s).mkdir(parents=True)
    cats = json.loads((ROOT / "dataset/train/_annotations.coco.json").read_text())["categories"]
    out = {s: Coco(cats) for s in ("train", "valid", "test")}

    names = [(n.split(":")[0], int(n.split(":")[1]) if ":" in n else 1) for n in a.real]
    test_name = a.test or names[0][0]
    for name, repeat in names:
        base = ROOT / "real" / name
        clean = None  # 没有 _log.json 的数据集（比如剪贴增强的）全部都用
        if (base / "_log.json").exists():
            log = json.loads((base / "_log.json").read_text())
            clean = {f"{name}_{int(k[2:6]):04d}.jpg" for k, v in log.items() if v["projected"] == 0 and v["occluded"] == 0}
        for split in ("train", "valid"):
            coco = json.loads((base / split / "_annotations.coco.json").read_text())
            anns = by_image(coco)
            n = 0
            for img in coco["images"]:
                if clean is not None and img["file_name"] not in clean:
                    continue
                n += 1
                if split == "train":
                    for r in range(repeat):
                        out["train"].add(base / split, img, anns.get(img["id"], []), OUT / "train",
                                         f"r{r}_{img['file_name']}")
                else:
                    out["valid"].add(base / split, img, anns.get(img["id"], []), OUT / "valid")
                    if name == test_name:
                        out["test"].add(base / split, img, anns.get(img["id"], []), OUT / "test")
            print(f"{name} {split}: {n} 张")

    for split, k in (("train", a.synth), ("valid", a.synth_valid)):
        coco = json.loads((ROOT / "dataset" / split / "_annotations.coco.json").read_text())
        anns = by_image(coco)
        for img in random.sample(coco["images"], k):
            out[split].add(ROOT / "dataset" / split, img, anns.get(img["id"], []), OUT / split)

    for s, c in out.items():
        (OUT / s / "_annotations.coco.json").write_text(json.dumps(c.d))
        print(f"{s}: {len(c.d['images'])} 张图, {len(c.d['annotations'])} 个框")


if __name__ == "__main__":
    main()
