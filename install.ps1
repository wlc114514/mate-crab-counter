# 数螃蟹 安装脚本：找/装 Python -> 建 .venv -> 装 torch（有 N 卡装 CUDA 版，否则 CPU 版）和 rfdetr
# -> 用示例图自检 -> 桌面建「QGC + 数螃蟹」快捷方式。
# 用法：双击 install.bat；强制装 CPU 版：install.bat -Cpu；不建快捷方式：install.bat -NoShortcut
param([switch]$Cpu, [switch]$NoShortcut)

$ErrorActionPreference = "Continue"  # 原生命令写 stderr 不算失败，靠 $LASTEXITCODE 判断
$Root = $PSScriptRoot
Set-Location $Root
[Console]::OutputEncoding = [Text.Encoding]::UTF8
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

$TORCH = "torch==2.11.0", "torchvision==0.26.0"
$PKGS = "rfdetr==1.12.0", "opencv-python-headless==5.0.0.93", "pillow"

function Step($msg) { Write-Host ""; Write-Host "== $msg" -ForegroundColor Cyan }
function Fail($msg) { Write-Host ""; Write-Host "安装失败：$msg" -ForegroundColor Red; exit 1 }

# ---- 1. Python 3.10 ~ 3.13
function Test-Py($exe, $extra) {
    try {
        $v = & $exe @extra -c "import sys, tkinter; print('%d.%d' % sys.version_info[:2])" 2>$null
        if ($LASTEXITCODE -eq 0 -and $v -match '^3\.(1[0-3])$') { return $true }
    } catch {}
    return $false
}

function Find-Py {
    foreach ($ver in "3.12", "3.13", "3.11", "3.10") {
        if ((Get-Command py -ErrorAction SilentlyContinue) -and (Test-Py "py" @("-$ver"))) {
            return (& py "-$ver" -c "import sys; print(sys.executable)")
        }
    }
    foreach ($n in "312", "313", "311", "310") {
        foreach ($base in "$env:LOCALAPPDATA\Programs\Python", "$env:ProgramFiles") {
            $p = Join-Path $base "Python$n\python.exe"
            if ((Test-Path $p) -and (Test-Py $p @())) { return $p }
        }
    }
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if ($cmd -and $cmd.Source -notlike "*WindowsApps*" -and (Test-Py $cmd.Source @())) { return $cmd.Source }
    return $null
}

Step "检查 Python"
$Py = Find-Py
if (-not $Py) {
    if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
        Start-Process "https://www.python.org/downloads/release/python-31210/"
        Fail "没找到 Python 3.10~3.13，也没有 winget。请从打开的网页安装 Python 3.12（勾选 Add python.exe to PATH），再重新运行 install.bat"
    }
    Write-Host "没找到可用的 Python，用 winget 安装 Python 3.12（只装给当前用户）…"
    winget install -e --id Python.Python.3.12 --scope user --silent --accept-package-agreements --accept-source-agreements
    $Py = Find-Py
    if (-not $Py) { Fail "Python 装完了但没找到，请重新运行 install.bat；还不行就手动从 python.org 安装 Python 3.12" }
}
Write-Host "使用 Python：$Py"

# ---- 2. 虚拟环境
$VPy = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $VPy)) {
    Step "创建虚拟环境 .venv"
    & $Py -m venv (Join-Path $Root ".venv")
    if ($LASTEXITCODE -ne 0) { Fail "创建 .venv 失败" }
}

# ---- 3. 依赖
$gpu = $false
if (-not $Cpu -and (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
    & nvidia-smi -L 2>$null | Out-Null
    $gpu = ($LASTEXITCODE -eq 0)
}
$index = if ($gpu) { "https://download.pytorch.org/whl/cu128" } else { "https://download.pytorch.org/whl/cpu" }
Step ("安装 PyTorch（" + $(if ($gpu) { "NVIDIA 显卡 CUDA 版，约 3 GB" } else { "CPU 版，约 200 MB" }) + "）")
& $VPy -m pip install --disable-pip-version-check @TORCH --index-url $index
if ($LASTEXITCODE -ne 0) { Fail "PyTorch 安装失败（检查网络后重新运行 install.bat）" }

Step "安装 rfdetr 等依赖"
& $VPy -m pip install --disable-pip-version-check @PKGS @TORCH --extra-index-url $index
if ($LASTEXITCODE -ne 0) { Fail "依赖安装失败（检查网络后重新运行 install.bat）" }

# ---- 4. 自检
Step "自检：用示例图数一次（第一次运行要加载模型，稍等）"
$W = Join-Path $Root "runs\ft2\checkpoint_best_total.pth"
if (-not (Test-Path $W)) { Fail "缺少权重文件 runs\ft2\checkpoint_best_total.pth，压缩包可能没解压完整" }
$out = & $VPy (Join-Path $Root "scripts\count.py") (Join-Path $Root "assets\real\pool_01.png") 2>&1 | ForEach-Object { "$_" }
$line = $out | Where-Object { $_ -like "pool_01*" }
if ($line) { Write-Host $line } else { $out | Select-Object -Last 15 | Write-Host; Fail "自检没跑通（上面是最后的输出）" }
if ($line -notmatch "青蟹 4 只") { Write-Host "注意：示例图应该是 4 只青蟹，结果不一致" -ForegroundColor Yellow }
& $VPy -c "import torch; print('计算设备：' + ('显卡 ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'))"

# ---- 5. 桌面快捷方式
if (-not $NoShortcut) {
    Step "创建桌面快捷方式「QGC + 数螃蟹」"
    $qgc = & $VPy -c "import sys; sys.path.insert(0, r'$Root\scripts'); import overlay; print(overlay.find_qgc() or '')"
    $desk = [Environment]::GetFolderPath("Desktop")
    $s = (New-Object -ComObject WScript.Shell).CreateShortcut((Join-Path $desk "QGC + 数螃蟹.lnk"))
    $s.TargetPath = Join-Path $Root ".venv\Scripts\pythonw.exe"
    $s.Arguments = "`"$Root\scripts\overlay.py`" --with-qgc"
    $s.WorkingDirectory = $Root
    $s.Description = "打开 QGroundControl，并在上面加一个「数螃蟹」按钮"
    if ($qgc) { $s.IconLocation = "$qgc,0" }
    $s.Save()
    if ($qgc) { Write-Host "找到 QGC：$qgc" } else { Write-Host "没找到 QGC：第一次双击快捷方式时会让你选 QGroundControl.exe 的位置" -ForegroundColor Yellow }
}

# ---- 6. 登记识别服务，给「QGroundControl Vision」（魔改版 QGC）自动拉起用
Step "登记识别服务（QGroundControl Vision 会自动启动它）"
$cmd = "`"$Root\.venv\Scripts\pythonw.exe`" `"$Root\scripts\vision_server.py`" --exit-with-parent"
New-Item -Path "HKCU:\Software\QGC-Vision" -Force | Out-Null
Set-ItemProperty -Path "HKCU:\Software\QGC-Vision" -Name "ServerCommand" -Value $cmd
Write-Host $cmd

Write-Host ""
Write-Host "安装完成。双击桌面的「QGC + 数螃蟹」使用；只开按钮可以双击 start_crab_button.bat。" -ForegroundColor Green
