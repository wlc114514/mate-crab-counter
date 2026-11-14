"""给 QGC（视觉识别版）用的本地识别服务：数青蟹。

QGC 点「识别」按钮后取几秒视频帧，POST 到这里，这里每帧数一次、投票，返回结果。
任何模型只要实现同样的两个接口，就能接到 QGC 上。

接口（JSON，UTF-8）：
  GET  /health
       -> {"ok": true, "ready": true/false, "name": "青蟹计数", "api": 1}
  POST /analyze
       请求 {"frames": ["<base64 JPEG>", ...]}          # 按时间顺序，至少 1 帧
       返回 {"ok": true,
             "label": "青蟹", "count": 4,               # 大数字显示 count，label 是它的名字
             "text": "青蟹 4 只",                        # 一行结论
             "detail": "15 帧投票：4 只 ×15",             # 小字说明
             "image": "<base64 JPEG>",                   # 结果图（可选）
             "boxes": [{"x":0.1,"y":0.2,"w":0.05,"h":0.04,"label":"green","score":0.93}]}  # 相对原始帧 0~1（可选）
       出错 {"ok": false, "error": "原因"}

用法：
  python scripts/vision_server.py [--port 8765] [--weights ...] [--exit-with-parent]
  --exit-with-parent：标准输入被关掉（QGC 退出或崩溃）时自己退出，QGC 拉起服务时用
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import threading
import time
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from board import find_board  # noqa: E402
from count import ROOT, detect_image, load, vote  # noqa: E402

NAME = "青蟹计数"
GREEN = (90, 220, 61)  # BGR


class Engine:
    def __init__(self, weights: str, thr: float):
        self.weights, self.thr = weights, thr
        self.model = None
        self.error = None
        self.lock = threading.Lock()
        threading.Thread(target=self._load, daemon=True).start()

    def _load(self):
        try:
            self.model = load(self.weights)
            print(f"模型已加载：{self.weights}", flush=True)
        except Exception as e:  # noqa: BLE001
            self.error = f"模型加载失败：{e}"
            print(self.error, flush=True)

    def analyze(self, jpegs: list[bytes]) -> dict:
        if self.model is None:
            return {"ok": False, "error": self.error or "模型还在加载，稍等几秒再试"}
        rgbs = []
        for b in jpegs:
            bgr = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
            if bgr is not None:
                rgbs.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        if not rgbs:
            return {"ok": False, "error": "没收到能解码的画面"}
        t0 = time.time()
        with self.lock:  # 一次只跑一个请求
            results = []
            for rgb in rgbs:
                _, det, names = detect_image(self.model, Image.fromarray(rgb), self.thr)
                results.append((names.count("green"), rgb, det, names))
        counts = [r[0] for r in results]
        best = vote(counts)
        _, rgb, det, names = max((r for r in results if r[0] == best),
                                 key=lambda r: float(r[2].confidence.min()) if len(r[2]) else 0.0)
        h, w = rgb.shape[:2]
        boxes = [{"x": float(x0 / w), "y": float(y0 / h), "w": float((x1 - x0) / w), "h": float((y1 - y0) / h),
                  "label": n, "score": float(s)}
                 for (x0, y0, x1, y1), n, s in zip(det.xyxy, names, det.confidence) if n == "green"]
        # 结果图：只框青蟹，裁到板子附近
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        box = find_board(bgr, margin=0.03)
        shown_w = (box[2] - box[0]) if box is not None else w
        t = max(3, round(shown_w / 130))
        for (x0, y0, x1, y1), n in zip(det.xyxy.astype(int), names):
            if n == "green":
                cv2.rectangle(bgr, (x0, y0), (x1, y1), GREEN, t)
        if box is not None:
            bgr = bgr[box[1]:box[3], box[0]:box[2]]
        ok, jpg = cv2.imencode(".jpg", np.ascontiguousarray(bgr), [cv2.IMWRITE_JPEG_QUALITY, 88])
        dist = "，".join(f"{k} 只 ×{v}" for k, v in sorted(Counter(counts).items()))
        print(f"识别：{len(rgbs)} 帧 -> {best} 只（{dist}），{time.time() - t0:.1f} 秒", flush=True)
        return {"ok": True, "label": "青蟹", "count": best, "text": f"青蟹 {best} 只",
                "detail": f"{len(rgbs)} 帧投票：{dist}",
                "image": base64.b64encode(jpg.tobytes()).decode() if ok else "",
                "boxes": boxes}


def make_handler(engine: Engine):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, obj: dict):
            body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            if self.path.rstrip("/") == "/health":
                self._send(200, {"ok": True, "ready": engine.model is not None, "name": NAME, "api": 1,
                                 "error": engine.error})
            else:
                self._send(404, {"ok": False, "error": "没有这个接口"})

        def do_POST(self):  # noqa: N802
            if self.path.rstrip("/") != "/analyze":
                return self._send(404, {"ok": False, "error": "没有这个接口"})
            try:
                n = int(self.headers.get("Content-Length", 0))
                req = json.loads(self.rfile.read(n))
                frames = [base64.b64decode(f) for f in req.get("frames", [])]
                self._send(200, engine.analyze(frames))
            except Exception as e:  # noqa: BLE001
                self._send(200, {"ok": False, "error": f"识别出错：{e}"})

        def log_message(self, fmt, *args):  # 只打印出错的请求
            if args and str(args[1]) not in ("200",):
                print(fmt % args, flush=True)

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--weights", default=str(ROOT / "runs/ft2/checkpoint_best_total.pth"))
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--exit-with-parent", action="store_true")
    a = ap.parse_args()
    if sys.stdout is None or sys.stderr is None:  # pythonw：输出写日志
        log = ROOT / "out" / "vision_server.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        sys.stdout = sys.stderr = log.open("a", encoding="utf-8", buffering=1)
    else:
        for s in (sys.stdout, sys.stderr):
            try:
                s.reconfigure(encoding="utf-8")
            except Exception:  # noqa: BLE001
                pass
    if a.exit_with_parent:
        def watch():
            try:
                while sys.stdin.buffer.read(1024):
                    pass
            except Exception:  # noqa: BLE001
                pass
            os._exit(0)
        threading.Thread(target=watch, daemon=True).start()
    try:
        srv = ThreadingHTTPServer((a.host, a.port), make_handler(Engine(a.weights, a.thr)))
    except OSError as e:
        print(f"端口 {a.port} 被占用（可能已经有一个服务在跑）：{e}", flush=True)
        sys.exit(2)
    print(f"{NAME} 服务：http://{a.host}:{a.port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
