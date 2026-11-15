"""吸附在 QGroundControl 窗口上的「数螃蟹」悬浮按钮。

点按钮 -> 取 3 秒画面（先连摄像头视频流，连不上就截 QGC 窗口）-> 每帧数一次、投票
-> 弹出结果：青蟹数量 + 只框青蟹的图。

用法：
  pythonw scripts/overlay.py --with-qgc           # 同时打开 QGC，QGC 关了自己退出（桌面快捷方式用这个）
  pythonw scripts/overlay.py                      # 只开按钮（视频流地址从 QGC 设置里读）
  python  scripts/overlay.py --source 录像.mkv --at 250      # 用录像当视频流来测试
  python  scripts/overlay.py --source screen                 # 只用截屏
按钮可以拖动，位置会记住（相对 QGC 窗口）。点结果面板关闭。
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import queue
import re
import sys
import threading
import time
import tkinter as tk
from collections import Counter
from ctypes import wintypes
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out" / "overlay"
STATE = OUT / "state.json"
QGC_INI = Path(os.environ.get("APPDATA", "")) / "QGroundControl.org" / "QGroundControl.ini"

BG, FG, ACCENT, DIM = "#1b1f23", "#ffffff", "#3ddc5a", "#9aa4ad"
FONT = "Microsoft YaHei UI"

user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32
dwmapi = ctypes.windll.dwmapi


# ---------------------------------------------------------------- Windows 窗口工具

def exe_of(hwnd) -> str:
    pid = wintypes.DWORD()
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    h = kernel32.OpenProcess(0x1000, False, pid.value)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not h:
        return ""
    buf = ctypes.create_unicode_buffer(520)
    n = wintypes.DWORD(520)
    ok = kernel32.QueryFullProcessImageNameW(h, 0, buf, ctypes.byref(n))
    kernel32.CloseHandle(h)
    return os.path.basename(buf.value).lower() if ok else ""


def find_window(exe: str, title_part: str | None):
    """找目标窗口：优先按进程名，其次按标题里的关键字。返回 hwnd 或 None。"""
    found = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def cb(hwnd, _):
        if not user32.IsWindowVisible(hwnd) or user32.IsIconic(hwnd):
            return True
        n = user32.GetWindowTextLengthW(hwnd)
        if n == 0:
            return True
        buf = ctypes.create_unicode_buffer(n + 1)
        user32.GetWindowTextW(hwnd, buf, n + 1)
        if title_part:
            if title_part.lower() in buf.value.lower():
                found.append(hwnd)
        elif exe_of(hwnd) == exe:
            found.append(hwnd)
        return True

    user32.EnumWindows(cb, 0)
    if not found:
        return None
    # 取面积最大的那个（主窗口）
    return max(found, key=lambda h: (lambda r: (r[2] - r[0]) * (r[3] - r[1]))(frame_rect(h)))


def exe_has_window(exe: str) -> bool:
    """这个程序还有没有窗口（最小化的也算）。用来判断 QGC 是不是真的关了。"""
    hit = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def cb(hwnd, _):
        if user32.GetWindowTextLengthW(hwnd) and exe_of(hwnd) == exe:
            hit.append(hwnd)
            return False
        return True

    user32.EnumWindows(cb, 0)
    return bool(hit)


def running_exe_path(exe: str) -> str | None:
    """正在运行的某个程序的完整路径（通过它的窗口找进程）。"""
    found = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def cb(hwnd, _):
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        h = kernel32.OpenProcess(0x1000, False, pid.value)
        if h:
            buf = ctypes.create_unicode_buffer(520)
            n = wintypes.DWORD(520)
            if kernel32.QueryFullProcessImageNameW(h, 0, buf, ctypes.byref(n)) and os.path.basename(buf.value).lower() == exe:
                found.append(buf.value)
            kernel32.CloseHandle(h)
        return not found

    user32.EnumWindows(cb, 0)
    return found[0] if found else None


def _exe_in(folder: Path) -> Path | None:
    for p in (folder / "QGroundControl.exe", folder / "bin" / "QGroundControl.exe"):
        if p.is_file():
            return p
    return None


def find_qgc() -> str | None:
    """自动找 QGC：正在运行的 → 常见安装目录 → 注册表安装记录 → 开始菜单快捷方式。"""
    p = running_exe_path("qgroundcontrol.exe")
    if p:
        return p
    env = os.environ.get
    for base in (env("ProgramFiles"), env("ProgramW6432"), env("ProgramFiles(x86)"), env("LOCALAPPDATA") and Path(env("LOCALAPPDATA")) / "Programs"):
        if base and (exe := _exe_in(Path(base) / "QGroundControl")):
            return str(exe)
    try:
        import winreg
        keys = [(winreg.HKEY_LOCAL_MACHINE, r"Software\Microsoft\Windows\CurrentVersion\Uninstall"),
                (winreg.HKEY_LOCAL_MACHINE, r"Software\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall"),
                (winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Uninstall")]
        for hive, path in keys:
            try:
                root = winreg.OpenKey(hive, path)
            except OSError:
                continue
            for i in range(winreg.QueryInfoKey(root)[0]):
                try:
                    sub = winreg.OpenKey(root, winreg.EnumKey(root, i))
                    if "qgroundcontrol" not in str(winreg.QueryValueEx(sub, "DisplayName")[0]).lower():
                        continue
                except OSError:
                    continue
                for name in ("InstallLocation", "DisplayIcon", "UninstallString"):
                    try:
                        v = str(winreg.QueryValueEx(sub, name)[0]).split(",")[0].strip().strip('"')
                    except OSError:
                        continue
                    if v and (exe := _exe_in(Path(v) if Path(v).is_dir() else Path(v).parent)):
                        return str(exe)
    except Exception:  # noqa: BLE001
        pass
    try:  # 开始菜单快捷方式：用 PowerShell 解析 .lnk 的目标
        import subprocess
        ps = ("$s=New-Object -ComObject WScript.Shell;"
              "Get-ChildItem \"$env:ProgramData\\Microsoft\\Windows\\Start Menu\\Programs\",\"$env:APPDATA\\Microsoft\\Windows\\Start Menu\\Programs\""
              " -Recurse -Filter *QGroundControl*.lnk -EA SilentlyContinue | ForEach-Object { $s.CreateShortcut($_.FullName).TargetPath }")
        out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True, timeout=15,
                             creationflags=0x08000000).stdout
        for line in out.splitlines():
            if line.strip().lower().endswith("qgroundcontrol.exe") and Path(line.strip()).is_file():
                return line.strip()
    except Exception:  # noqa: BLE001
        pass
    return None


def load_state() -> dict:
    try:
        return json.loads(STATE.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def save_state(**kw) -> None:
    s = load_state()
    s.update(kw)
    OUT.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(s, ensure_ascii=False), encoding="utf-8")


def resolve_qgc(given: str | None, root: tk.Tk) -> str | None:
    """确定 QGC 路径：命令行给的 → 上次记住的 → 自动找 → 让用户选一次（选完记住）。"""
    if given and Path(given).is_file():
        return given
    saved = load_state().get("qgc_path")
    if saved and Path(saved).is_file():
        return saved
    p = find_qgc()
    if not p:
        from tkinter import filedialog, messagebox
        messagebox.showinfo("数螃蟹", "没找到 QGroundControl，请选一下 QGroundControl.exe 的位置（只需选一次）。", parent=root)
        p = filedialog.askopenfilename(parent=root, title="选择 QGroundControl.exe",
                                       filetypes=[("QGroundControl", "QGroundControl.exe"), ("程序", "*.exe")]) or None
    if p:
        save_state(qgc_path=str(Path(p)))
    return p


def frame_rect(hwnd):
    r = wintypes.RECT()
    if dwmapi.DwmGetWindowAttribute(hwnd, 9, ctypes.byref(r), ctypes.sizeof(r)) != 0:  # 不含阴影的外框
        user32.GetWindowRect(hwnd, ctypes.byref(r))
    return r.left, r.top, r.right, r.bottom


def client_rect(hwnd):
    r = wintypes.RECT()
    user32.GetClientRect(hwnd, ctypes.byref(r))
    p = wintypes.POINT(0, 0)
    user32.ClientToScreen(hwnd, ctypes.byref(p))
    return p.x, p.y, p.x + r.right, p.y + r.bottom


def no_activate(win: tk.Toplevel | tk.Tk) -> None:
    """点悬浮窗不抢焦点，键盘和手柄输入留在 QGC。"""
    win.update_idletasks()
    hwnd = user32.GetParent(win.winfo_id()) or win.winfo_id()
    style = user32.GetWindowLongW(hwnd, -20)
    user32.SetWindowLongW(hwnd, -20, style | 0x08000000 | 0x00000080)  # NOACTIVATE | TOOLWINDOW


def foreground_is(hwnd) -> bool:
    fg = user32.GetForegroundWindow()
    if fg == hwnd:
        return True
    pid = wintypes.DWORD()
    user32.GetWindowThreadProcessId(fg, ctypes.byref(pid))
    return pid.value == os.getpid()


# ---------------------------------------------------------------- 取画面

def rtsp_url_from_qgc() -> str | None:
    try:
        m = re.search(r"^rtspUrl=(.+)$", QGC_INI.read_text(encoding="utf-8", errors="ignore"), re.M)
        return m.group(1).strip() if m else None
    except OSError:
        return None


def grab_stream(src: str, at: float, seconds: float, fps: float, timeout: float = 5.0):
    """从视频流或录像取帧。连不上（超时）返回 []。"""
    import cv2

    live = "://" in src
    box: dict = {}

    def opener():
        if live:
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|timeout;4000000"
        cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        box["cap"] = cap if cap.isOpened() else None

    t = threading.Thread(target=opener, daemon=True)
    t.start()
    t.join(timeout if live else 30)
    cap = box.get("cap")
    if cap is None:
        return []
    if not live and at > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, at * 1000)
    native = cap.get(cv2.CAP_PROP_FPS)
    if not native or native > 120:
        native = 30.0
    step = max(1, round(native / fps))
    frames, i, t0 = [], 0, time.time()
    want = int(seconds * fps)
    while len(frames) < want and time.time() - t0 < seconds + timeout:
        ok, f = cap.read()
        if not ok:
            break
        if i % step == 0:
            frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        i += 1
    cap.release()
    return frames


def grab_screen(hwnd, seconds: float, fps: float):
    import numpy as np
    from PIL import ImageGrab

    frames = []
    for _ in range(int(seconds * fps)):
        t0 = time.time()
        frames.append(np.asarray(ImageGrab.grab(bbox=client_rect(hwnd), all_screens=True).convert("RGB")))
        time.sleep(max(0.0, 1 / fps - (time.time() - t0)))
    return frames


# ---------------------------------------------------------------- 识别

def analyse(model, frames, thr: float):
    """返回 (青蟹数, 各帧计数, 只框青蟹的结果图 PIL)。"""
    import cv2
    import numpy as np
    from PIL import Image

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from board import find_board
    from count import detect_image, vote

    results = []
    for rgb in frames:
        _, det, names = detect_image(model, Image.fromarray(rgb), thr)
        results.append((names.count("green"), rgb, det, names))
    counts = [r[0] for r in results]
    best = vote(counts)
    _, rgb, det, names = max((r for r in results if r[0] == best),
                             key=lambda r: float(r[2].confidence.min()) if len(r[2]) else 0.0)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    box = find_board(bgr, margin=0.03)
    shown_w = (box[2] - box[0]) if box is not None else bgr.shape[1]
    t = max(3, round(shown_w / 130))  # 面板里会缩到 460 宽，框要够粗
    for (x0, y0, x1, y1), n in zip(det.xyxy.astype(int), names):
        if n == "green":
            cv2.rectangle(bgr, (x0, y0), (x1, y1), (90, 220, 61), t)
    if box is not None:
        bgr = bgr[box[1]:box[3], box[0]:box[2]]
    return best, counts, Image.fromarray(cv2.cvtColor(np.ascontiguousarray(bgr), cv2.COLOR_BGR2RGB))


# ---------------------------------------------------------------- 界面

class App:
    def __init__(self, a):
        self.a = a
        OUT.mkdir(parents=True, exist_ok=True)
        self.q: queue.Queue = queue.Queue()
        self.model = None
        self.busy = False
        self.hwnd = None
        self.rel = self._load_rel()
        self.drag = None
        self.panel = None
        self.photo = None

        self.root = tk.Tk()
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        self.root.configure(bg=ACCENT)
        self.btn = tk.Label(self.root, text="加载中…", bg=BG, fg=DIM, font=(FONT, 12, "bold"), padx=18, pady=8)
        self.btn.pack(padx=2, pady=2)
        self.btn.bind("<ButtonPress-1>", self._press)
        self.btn.bind("<B1-Motion>", self._motion)
        self.btn.bind("<ButtonRelease-1>", self._release)
        self.btn.bind("<Button-3>", lambda e: self.root.destroy())  # 右键退出
        no_activate(self.root)

        self.seen_qgc, self.gone_ticks, self.t0 = False, 0, time.time()
        if a.with_qgc and not exe_has_window("qgroundcontrol.exe"):
            import subprocess
            qgc = resolve_qgc(a.qgc_path, self.root)
            if not qgc:  # 用户取消了选择：没法带起 QGC，直接退出
                self.root.after(0, self.root.destroy)
                return
            subprocess.Popen([qgc], cwd=str(Path(qgc).parent))
        threading.Thread(target=self._load_model, daemon=True).start()
        self.root.after(100, self._tick)
        if a.selftest:
            self.root.after(500, self._selftest_wait)

    # ---- 状态
    def _load_rel(self):
        rel = load_state().get("rel")
        return tuple(rel) if rel else None

    def _save_rel(self):
        save_state(rel=self.rel)

    def _load_model(self):
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from count import load
            self.model = load(self.a.weights)
            self.q.put(("ready", None))
        except Exception as e:  # noqa: BLE001
            self.q.put(("error", f"模型加载失败：{e}"))

    # ---- 跟随 QGC
    def _tick(self):
        try:
            while True:
                kind, data = self.q.get_nowait()
                getattr(self, f"_on_{kind}")(data)
        except queue.Empty:
            pass
        self.hwnd = find_window("qgroundcontrol.exe", self.a.window_title)
        if self.hwnd and self.drag is None:
            l, t, r, b = frame_rect(self.hwnd)
            bw, bh = self.root.winfo_width(), self.root.winfo_height()
            scale = (user32.GetDpiForWindow(self.hwnd) or 96) / 96
            rel = self.rel or ((r - l - bw) // 2, round(76 * scale))  # 默认：顶部工具栏下方居中
            x = min(max(l, l + rel[0]), r - bw)
            y = min(max(t, t + rel[1]), b - bh)
            self.root.geometry(f"+{x}+{y}")
            show = foreground_is(self.hwnd)
        elif self.hwnd is None and self.drag is None:
            sw = self.root.winfo_screenwidth()
            self.root.geometry(f"+{sw - self.root.winfo_width() - 40}+60")  # QGC 没开：停在屏幕右上角
            show = not self.a.with_qgc  # 跟 QGC 一起启动的模式下，QGC 最小化或还没出来时先藏着
        else:
            show = True
        if self.a.with_qgc and not self.a.selftest:
            if self.hwnd:
                self.seen_qgc = True
            alive = bool(self.hwnd) or exe_has_window("qgroundcontrol.exe")
            self.gone_ticks = 0 if alive else self.gone_ticks + 1
            # QGC 关了（连续 2 秒没有窗口），或者 90 秒都没等到 QGC：退出
            if (self.seen_qgc and self.gone_ticks >= 8) or (not self.seen_qgc and time.time() - self.t0 > 90):
                self.root.destroy()
                return
        if show or self.a.selftest:
            self.root.deiconify()
            self.root.attributes("-topmost", True)
            if self.panel:
                self.panel.deiconify()
        else:  # QGC 不在最前面：按钮和结果先藏起来，切回来再显示
            self.root.withdraw()
            if self.panel:
                self.panel.withdraw()
        self.root.after(250, self._tick)

    # ---- 拖动 / 点击
    def _press(self, e):
        self.drag = {"x0": e.x_root, "y0": e.y_root, "wx": self.root.winfo_x(), "wy": self.root.winfo_y(), "moved": False}

    def _motion(self, e):
        d = self.drag
        if d is None:
            return
        dx, dy = e.x_root - d["x0"], e.y_root - d["y0"]
        if abs(dx) + abs(dy) > 6:
            d["moved"] = True
        if d["moved"]:
            self.root.geometry(f"+{d['wx'] + dx}+{d['wy'] + dy}")

    def _release(self, _e):
        d, self.drag = self.drag, None
        if d is None:
            return
        if d["moved"]:
            if self.hwnd:
                l, t, _, _ = frame_rect(self.hwnd)
                self.rel = (self.root.winfo_x() - l, self.root.winfo_y() - t)
                self._save_rel()
        else:
            self.start()

    # ---- 识别流程
    def start(self):
        if self.busy or self.model is None:
            return
        self.busy = True
        self._close_panel()
        self.btn.configure(text="识别中…", fg=DIM)
        threading.Thread(target=self._work, daemon=True).start()

    def _work(self):
        a = self.a
        try:
            frames, via = [], ""
            src = a.source or rtsp_url_from_qgc()
            if src and src != "screen":
                frames = grab_stream(src, a.at, a.seconds, a.fps)
                via = "视频流" if "://" in src else "录像"
            if not frames:
                if not self.hwnd:
                    raise RuntimeError("连不上视频流，也没找到 QGC 窗口")
                frames, via = grab_screen(self.hwnd, a.seconds, a.fps), "截屏"
            best, counts, img = analyse(self.model, frames, a.thr)
            self.q.put(("result", (best, counts, img, via)))
        except Exception as e:  # noqa: BLE001
            self.q.put(("error", str(e)))

    def _on_ready(self, _):
        self.btn.configure(text="数螃蟹", fg=FG)

    def _on_error(self, msg):
        self.busy = False
        self.btn.configure(text="数螃蟹" if self.model else "出错", fg=FG)
        self._show_panel(None, msg, None)
        (OUT / "overlay.log").open("a", encoding="utf-8").write(f"{datetime.now():%F %T} ERROR {msg}\n")

    def _on_result(self, data):
        best, counts, img, via = data
        self.busy = False
        self.btn.configure(text="数螃蟹", fg=FG)
        votes = " · ".join(f"{k}×{v}" for k, v in sorted(Counter(counts).items(), key=lambda kv: -kv[1]))
        self._show_panel(best, f"{via} · {len(counts)} 帧 · {votes}", img)
        stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
        img.save(OUT / f"{stamp}_green{best}.jpg", quality=90)
        (OUT / "overlay.log").open("a", encoding="utf-8").write(f"{stamp} green={best} via={via} counts={counts}\n")
        self.last = (best, counts, via)

    # ---- 结果面板
    def _close_panel(self):
        if self.panel is not None:
            self.panel.destroy()
            self.panel = None

    def _show_panel(self, best, sub, img):
        from PIL import ImageTk

        self._close_panel()
        p = tk.Toplevel(self.root)
        p.overrideredirect(True)
        p.attributes("-topmost", True)
        p.configure(bg=ACCENT if best is not None else "#e5534b")
        body = tk.Frame(p, bg=BG)
        body.pack(padx=2, pady=2)
        if best is not None:
            head = tk.Frame(body, bg=BG)
            head.pack(fill="x", padx=16, pady=(10, 0))
            tk.Label(head, text=str(best), bg=BG, fg=ACCENT, font=(FONT, 44, "bold")).pack(side="left")
            tk.Label(head, text="只青蟹", bg=BG, fg=FG, font=(FONT, 14)).pack(side="left", padx=(8, 0), pady=(18, 0))
        tk.Label(body, text=sub, bg=BG, fg=DIM if best is not None else FG, font=(FONT, 9), wraplength=440,
                 justify="left").pack(anchor="w", padx=16, pady=(0 if best is not None else 12, 8))
        if img is not None:
            w = 460
            im = img.resize((w, max(1, round(img.height * w / img.width))))
            self.photo = ImageTk.PhotoImage(im)
            tk.Label(body, image=self.photo, bg=BG, bd=0).pack(padx=10, pady=(0, 10))
        for wdg in [p, body, *body.winfo_children()]:
            wdg.bind("<Button-1>", lambda e: self._close_panel())  # 点一下关闭
        p.update_idletasks()
        bx, by = self.root.winfo_x(), self.root.winfo_y() + self.root.winfo_height() + 6
        x = bx + (self.root.winfo_width() - p.winfo_width()) // 2
        x = min(max(8, x), p.winfo_screenwidth() - p.winfo_width() - 8)
        y = min(by, p.winfo_screenheight() - p.winfo_height() - 48)
        p.geometry(f"+{x}+{y}")
        no_activate(p)
        self.panel = p

    # ---- 自测：模型好了自动点一次，截图后退出
    def _selftest_wait(self):
        if self.model is None:
            self.root.after(500, self._selftest_wait)
            return
        self.last = None
        self.start()
        self.root.after(500, self._selftest_done)

    def _selftest_done(self):
        if self.busy or self.panel is None:
            self.root.after(500, self._selftest_done)
            return
        from PIL import ImageGrab

        if self.hwnd:  # 自测时把目标窗口临时提到最前，保证截到的是它
            user32.SetWindowPos(self.hwnd, -1, 0, 0, 0, 0, 0x0013)
            self.root.attributes("-topmost", True); self.panel.attributes("-topmost", True)
            self.root.lift(); self.panel.lift()
        self.root.update()
        time.sleep(0.8)
        if self.hwnd:  # 目标窗口连同按钮一起截一张，看吸附位置
            ImageGrab.grab(bbox=frame_rect(self.hwnd), all_screens=True).save(OUT / "selftest_window.png")
        p = self.panel
        ImageGrab.grab(bbox=(p.winfo_x(), self.root.winfo_y(), p.winfo_x() + p.winfo_width(),
                             p.winfo_y() + p.winfo_height()), all_screens=True).save(OUT / "selftest_panel.png")
        if self.hwnd:
            user32.SetWindowPos(self.hwnd, -2, 0, 0, 0, 0, 0x0013)
        print("SELFTEST", getattr(self, "last", None))
        self.root.destroy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ROOT / "runs/ft3/checkpoint_best_total.pth"))
    ap.add_argument("--source", default=None, help="视频流地址 / 录像文件 / screen（默认读 QGC 设置里的 RTSP 地址）")
    ap.add_argument("--at", type=float, default=0.0)
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--fps", type=float, default=5.0)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--window-title", default=None, help="按标题找目标窗口（测试用）")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--with-qgc", action="store_true", help="同时打开 QGC；QGC 关闭时自己退出")
    ap.add_argument("--qgc-path", default=None, help="QGC 程序路径（不给就自动找，找不到会让你选一次）")
    a = ap.parse_args()
    if sys.stdout is None or sys.stderr is None:  # pythonw 没有控制台：输出写进日志，免得库打印时报错
        OUT.mkdir(parents=True, exist_ok=True)
        sys.stdout = sys.stderr = (OUT / "console.log").open("a", encoding="utf-8", buffering=1)
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:  # noqa: BLE001
        pass
    App(a).root.mainloop()


if __name__ == "__main__":
    main()
