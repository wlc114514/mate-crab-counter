# mate-crab-counter

MATE ROV 2026 NAVIGATOR Task 2.1：数样本板上的欧洲青蟹（European Green crab）。

模型是 RF-DETR Small，识别 `green` / `rock` / `jonah` 三类，只报 `green` 的数量。
训练数据全部由手册里的三张官方螃蟹图合成（规则写明只用这三张，只会缩放和旋转）。

## 结果

| 测试 | 结果 |
|---|---|
| 合成测试集 300 张，青蟹数量完全正确 | 300 / 300 |
| 合成测试集 mAP50-95 | 0.972 |
| 道具说明里的实拍样本板（4 只青蟹） | 4 只，正确 |

还没有在真实水下画面上验证过。

## 用法

权重在本仓库的 Release 里（`checkpoint_best_total.pth`），下载后放到 `runs/small/`。

```
python -m venv .venv
.venv\Scripts\pip install "rfdetr[train]==1.12.0"   # 需要先装好带 CUDA 的 torch
.venv\Scripts\python scripts\count.py 图片.jpg       # 输出青蟹数量，标框图存到 out/
```

## 重新训练

```
.venv\Scripts\python scripts\prep_crabs.py                      # 切出三张螃蟹图
.venv\Scripts\python scripts\make_dataset.py --workers 4        # 生成合成数据集（约 5 分钟）
.venv\Scripts\python scripts\train.py --epochs 30               # RTX 5060 上约 2 小时
.venv\Scripts\python scripts\count.py --eval                    # 合成测试集计数准确率
```

## 目录

- `rules/`：2026 NAVIGATOR 比赛手册和道具制作说明
- `assets/source/`：手册第 27 页的官方螃蟹图
- `assets/prop_photos/`：道具说明里的实拍样本板
- `scripts/`：素材处理、数据合成、训练、计数
