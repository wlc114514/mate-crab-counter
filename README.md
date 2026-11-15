# mate-crab-counter

MATE ROV 2026 NAVIGATOR Task 2.1：数样本板上的欧洲青蟹（European Green crab）。

模型是 RF-DETR Small，识别 `green` / `rock` / `jonah` 三类，只报 `green` 的数量。
先用手册里三张官方螃蟹图合成数据训练（`runs/small`），再用真实水池录像微调：`runs/ft2`（11-12 旧泳池），`runs/ft3`（加了 11-15 新泳池、新摄像头，当前默认）。

## 接到 QGroundControl Vision 使用

[QGroundControl Vision](https://github.com/wlc114514/qgc-vision) 是加了视觉识别接口的 QGC 4.4.4。装好它和本项目后，飞行界面左侧工具栏会多一个 **识别** 按钮：点一下，QGC 从自己的视频流里取 3 秒画面交给数螃蟹模型，几秒后在画面上方显示「青蟹 N 只」和只框青蟹的结果图。

### 1. 安装 QGroundControl Vision

从 [qgc-vision Releases](https://github.com/wlc114514/qgc-vision/releases) 下载 `QGroundControl-Vision-4.4.4-installer.exe`，双击安装。

- 装在单独的「QGroundControl Vision」目录，桌面和开始菜单各有一个快捷方式。
- 不覆盖、不卸载官方 QGroundControl，两个可以同时装着；它们**共用同一份设置**，官方 QGC 里配好的飞控连接和视频源直接就能用。

### 2. 安装数螃蟹模型

1. 从本项目 [Releases](https://github.com/wlc114514/mate-crab-counter/releases) 下载 `crab-counter-v1.2.zip`，解压到一个固定位置（比如 `D:\crab-counter`）。**别放在临时文件夹或下载目录里随手删掉**：QGC Vision 每次启动都会从这里运行识别服务。
2. 双击 `install.bat`。它会：
   - 没有 Python 3.10~3.13 时用 winget 自动安装 Python 3.12；
   - 建 `.venv`，有 NVIDIA 显卡装 CUDA 版 PyTorch，否则装 CPU 版（Intel / AMD 显卡的电脑都走 CPU）；
   - 用示例图自检（应输出「青蟹 4 只」）；
   - 把识别服务登记给 QGC Vision（注册表 `HKCU\Software\QGC-Vision\ServerCommand`）。
3. 需要联网，CPU 版大约 5 分钟。装好以后识别不需要联网。

文件夹挪了位置，重新运行一次 `install.bat` 就行。`install.bat -Cpu`：有 N 卡也强制装 CPU 版。

### 3. 确认视频

打开 QGroundControl Vision，飞行界面能看到 ROV 的画面就行。设置和官方 QGC 共用，官方 QGC 里有画面，这里就有。

注意 **ROV 连接和视频是分开设置的**：Application Settings → Comm Links 里的「ROV」（TCP `192.168.1.200:4019`）只传飞控数据，ROV 不会把视频地址告诉 QGC。视频要在 Application Settings → 常规 → 视频 里单独设：源选「RTSP Video Stream」，填 ROV 摄像头的 RTSP 地址（比如 `rtsp://192.168.1.10:554/stream1`）。

- 摄像头是 H.265 1080p 也没问题；RTSP 不通 UDP 时 QGC 会在几秒后自动改走 TCP，不用额外设置。
- 「识别」取的是 QGC 解码后的画面，所以只要 QGC 能播出来，不管视频走 TCP 还是 UDP、H.264 还是 H.265 都能识别。

### 4. 识别

1. 把 ROV 对准样本板，整块板子尽量都在画面里，别让爪子挡住。
2. 点左侧工具栏的 **识别**。按钮变成「识别中」，画面上方依次显示「取画面 n/15」→「识别中（15 帧）…」。
3. 出结果：大字显示青蟹数量，下面是多帧投票明细（比如「15 帧投票：4 只 ×15」）和只框青蟹的结果图。点面板关闭。

速度参考：RTX 5060 每帧 27 ms，点完 3 秒取画面后基本马上出结果；CPU（4 线程）每帧约 0.4 秒，取完画面还要再等 5~15 秒。

识别服务随 QGC Vision 自动启动、自动关闭，不用手动开。刚打开 QGC 时模型要加载一会儿，这时点识别会提示「模型还在加载，稍等几秒再试」。

### 设置

Application Settings → 常规 → 视觉识别：

| 设置 | 默认 | 说明 |
|---|---|---|
| 显示「识别」按钮 | 开 | |
| 服务状态 | | 已就绪 / 模型加载中… / 正在启动服务… / 连不上；旁边有「重启服务」 |
| 服务地址 | `http://127.0.0.1:8765` | |
| 启动命令 | 空 | 空的时候用 `install.bat` 登记的命令，一般不用填 |
| 每次取画面（秒） | 3 | 板子晃得厉害可以加长 |
| 每秒取几帧 | 5 | CPU 慢的电脑可以降到 2~3 |
| QGC 启动时自动启动识别服务，退出时关闭 | 开 | |

### 出问题时

| 现象 | 处理 |
|---|---|
| 左侧没有「识别」按钮 | 打开的是官方 QGroundControl，不是 QGroundControl Vision；或者设置里「显示识别按钮」被关了 |
| 「没有视频画面，先确认视频在播放」 | 先让飞行界面出画面（见第 3 步） |
| 「连不上识别服务」/ 服务状态「没有设置启动命令」或「启动失败」 | 没运行过 `install.bat`，或者装完以后挪了文件夹 → 重新运行 `install.bat`，然后在设置里点「重启服务」 |
| 「服务已退出（代码 …）」 | 看服务日志 `%TEMP%\qgc-vision\server.log` |
| 数量不对 | 看结果图里框了哪些：板子太小、太斜、被爪子挡住都会少数，靠近一点、正对着再识别一次 |

结果图保存在 `%TEMP%\qgc-vision\`。

## 不用 QGroundControl Vision 时

- **悬浮按钮**：`install.bat -Shortcut` 会在桌面建「QGC + 数螃蟹」快捷方式，打开（官方）QGroundControl 并在窗口上方挂一个「数螃蟹」按钮。点按钮 → 取 3 秒画面（先直连 QGC 设置里的 RTSP 视频流，5 秒连不上就截 QGC 窗口）→ 投票 → 显示结果。按钮可以拖动，右键退出；QGC 关掉后按钮自己退出。结果图存在 `out/overlay/`。只开按钮：双击 `start_crab_button.bat`。
- **命令行**：
  ```
  .venv\Scripts\python scripts\count.py 图片.jpg                      # 数图片，标框图存到 out/
  .venv\Scripts\python scripts\count.py --video 录像.mkv --at 200      # 录像第 200 秒起 3 秒，多帧投票
  .venv\Scripts\python scripts\count.py --video rtsp://192.168.1.10:554/stream1
  ```
- **识别服务单独运行**（调试或接别的程序）：`.venv\Scripts\python scripts\vision_server.py`，监听 `http://127.0.0.1:8765`，接口见 [qgc-vision 的 VISION.md](https://github.com/wlc114514/qgc-vision/blob/main/VISION.md)。

## 结果

| 测试 | ft2 | ft3（默认） |
|---|---|---|
| 合成测试集 300 张，青蟹数量完全正确 | 299 / 300 | 298 / 300 |
| 旧泳池（11-12）没参与训练的 30 秒，单帧 | 94.7% | 96.7% |
| 同上，3 秒多帧投票 | 10 / 10 | 10 / 10 |
| 新泳池 + 新摄像头（11-15）没参与训练的最后 15 秒，单帧 | 11.5% | 64.1% |
| 同上，3 秒多帧投票 | 0 / 5 | 5 / 5 |

新泳池的摄像头几乎拍不出红色，ft2 靠颜色分品种，把 rock / jonah 当成青蟹。ft3 在训练里加了新泳池的画面，又把旧数据去红、偏青、降饱和（`scripts/color_aug.py`），逼模型看形状和花纹。

局限：新泳池只有一段 64 秒录像、一种摆法，测试帧和训练帧来自同一段录像，成绩偏乐观；新摄像头下 rock / jonah 仍常互相认错（单帧三类全对 37%），只报青蟹数影响较小。ROV 爪子挡住板子时会少数。

## 重新训练

```
.venv\Scripts\pip install "rfdetr[train]==1.12.0"                 # 训练需要 train 依赖和 N 卡
.venv\Scripts\python scripts\prep_crabs.py                         # 切出三张螃蟹图
.venv\Scripts\python scripts\make_dataset.py --workers 4           # 合成数据集
.venv\Scripts\python scripts\train.py --epochs 30                  # 合成数据训练 -> runs/small
# 真实录像微调：抽帧 -> label_real.py 自动标注 -> copypaste.py 换摆法增强
# -> color_aug.py 颜色退化 -> make_ft_dataset.py 合并 -> train.py --weights runs/ft2/... --lr 5e-5 --out runs/ft3
.venv\Scripts\python scripts\count.py --eval                       # 合成测试集计数准确率
```

## 目录

- `rules/`：2026 NAVIGATOR 手册、道具说明、评分表，RANGER 手册
- `assets/`：官方螃蟹图、道具实拍、水下截图
- `scripts/`：`count.py` 计数，`board.py` 找样本板，`vision_server.py` 给 QGroundControl Vision 用的识别服务，`overlay.py` QGC 悬浮按钮，其余是数据合成 / 标注 / 训练 / 评估
- `install.bat` / `install.ps1`：安装脚本（装依赖、自检、登记识别服务）
