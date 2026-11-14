# mate-crab-counter

MATE ROV 2026 NAVIGATOR Task 2.1：数样本板上的欧洲青蟹（European Green crab）。

模型是 RF-DETR Small，识别 `green` / `rock` / `jonah` 三类，只报 `green` 的数量。
先用手册里三张官方螃蟹图合成数据训练（`runs/small`），再用真实水池录像微调（`runs/ft2`，当前默认）。

## 安装（推荐：用 Release 里的压缩包）

1. 从 Release 下载 `crab-counter-v1.1.zip`，解压到一个固定位置（比如 `D:\crab-counter`，别放在临时文件夹里）。
2. 双击 `install.bat`。它会：
   - 没有 Python 3.10~3.13 时用 winget 自动安装 Python 3.12；
   - 建 `.venv`，有 NVIDIA 显卡装 CUDA 版 PyTorch，否则装 CPU 版（Intel / AMD 显卡的电脑都走 CPU）；
   - 用示例图自检（应输出「青蟹 4 只」）；
   - 在桌面建「QGC + 数螃蟹」快捷方式。
3. 需要联网，CPU 版大约 5 分钟。装好以后识别不需要联网。

参数：`install.bat -Cpu` 有 N 卡也强制装 CPU 版，`install.bat -NoShortcut` 不建快捷方式。

速度参考：RTX 5060 每帧 27 ms；CPU 4 线程每帧约 0.4 秒。按一次按钮要算 15 帧，CPU 上要等 5~15 秒。

## 使用

- **QGC + 数螃蟹**（桌面快捷方式）：打开 QGroundControl，并在 QGC 窗口上方挂一个「数螃蟹」按钮。
  点按钮 → 取 3 秒画面（先直连 QGC 设置里的 RTSP 视频流，5 秒连不上就截 QGC 窗口）→ 每帧数一次、投票 → 显示青蟹数量和只框青蟹的图。
  按钮可以拖动，右键退出；QGC 关掉后按钮自己退出。结果图存在 `out/overlay/`。
  QGC 路径自动找（正在运行的 / 默认安装目录 / 注册表 / 开始菜单），找不到会让你选一次 `QGroundControl.exe` 并记住。
- 只开按钮：双击 `start_crab_button.bat`。
- 命令行：
  ```
  .venv\Scripts\python scripts\count.py 图片.jpg                      # 数图片，标框图存到 out/
  .venv\Scripts\python scripts\count.py --video 录像.mkv --at 200      # 录像第 200 秒起 3 秒，多帧投票
  .venv\Scripts\python scripts\count.py --video rtsp://192.168.1.10:554/stream1
  ```

## 结果（runs/ft2）

| 测试 | 结果 |
|---|---|
| 合成测试集 300 张，青蟹数量完全正确 | 299 / 300 |
| 真实水池录像，没参与训练的最后 30 秒，单帧 | 95.3% |
| 同上，3 秒多帧投票 | 27 / 27 |
| 换摆法（剪贴合成）验证集 | 96.3%（和训练用同一种剪贴方法，偏乐观） |
| 3 张水下截图 + 道具说明实拍板 | 全对 |

局限：真实数据只来自一个泳池、一套打印的螃蟹、一种灯光；换场地还没验证。ROV 爪子挡住板子时会少数。

## 重新训练

```
.venv\Scripts\pip install "rfdetr[train]==1.12.0"                 # 训练需要 train 依赖和 N 卡
.venv\Scripts\python scripts\prep_crabs.py                         # 切出三张螃蟹图
.venv\Scripts\python scripts\make_dataset.py --workers 4           # 合成数据集
.venv\Scripts\python scripts\train.py --epochs 30                  # 合成数据训练 -> runs/small
# 真实录像微调：抽帧 -> label_real.py 自动标注 -> copypaste.py 换摆法增强
# -> make_ft_dataset.py 合并 -> train.py --weights runs/small/... --lr 5e-5 --out runs/ft2
.venv\Scripts\python scripts\count.py --eval                       # 合成测试集计数准确率
```

## 目录

- `rules/`：2026 NAVIGATOR 手册、道具说明、评分表，RANGER 手册
- `assets/`：官方螃蟹图、道具实拍、水下截图
- `scripts/`：`count.py` 计数，`board.py` 找样本板，`overlay.py` QGC 悬浮按钮，其余是数据合成 / 标注 / 训练 / 评估
- `install.bat` / `install.ps1`：安装脚本
