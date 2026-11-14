"""训练 RF-DETR Small 识别三种螃蟹。

用法： python scripts/train.py [--epochs 30] [--resume runs/small/last.ckpt]
微调： python scripts/train.py --data dataset_ft --weights runs/small/checkpoint_best_total.pth --lr 5e-5 --epochs 10 --out runs/ft1
输出： runs/small/checkpoint_best_total.pth（推理用这个）
"""
import argparse
from pathlib import Path

from rfdetr import RFDETRSmall

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4)  # 有效 batch = 4 x 4 = 16，官方推荐
    ap.add_argument("--workers", type=int, default=2)  # 数据加载进程少一点，CPU 别吃满
    ap.add_argument("--out", default=str(ROOT / "runs/small"))
    ap.add_argument("--resume", default=None)
    ap.add_argument("--data", default=str(ROOT / "dataset"))
    ap.add_argument("--weights", default=None, help="从已有权重开始微调")
    ap.add_argument("--lr", type=float, default=1e-4)
    a = ap.parse_args()

    model = RFDETRSmall(pretrain_weights=a.weights) if a.weights else RFDETRSmall()
    model.train(
        dataset_dir=a.data,
        output_dir=a.out,
        epochs=a.epochs,
        batch_size=a.batch,
        grad_accum_steps=a.accum,
        num_workers=a.workers,
        lr=a.lr,
        resolution=512,
        early_stopping=True,
        early_stopping_patience=6,
        log_per_class_metrics=True,
        run_test=True,
        seed=0,
        resume=a.resume,
        progress_bar="tqdm",
    )


if __name__ == "__main__":  # Windows 下 DataLoader 多进程必须有这行
    main()
