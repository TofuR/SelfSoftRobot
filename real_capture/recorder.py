# recorder.py
"""六通道阀 + NDI + 相机 的同步采集核心（动作门控 / action-gated）。

设计要点
--------
- **动作门控采集**（核心）：动作每 `action_interval_s`（默认 0.2s）下发一次；下发后等
  `settle_s`（默认 0.19s）让软臂稳定，再到缓存里取**最新的一帧 + 一个 NDI 末端位姿**落盘。
  → 每拍产出一组 `(action_i, frame_i, ndi_i)` 同索引样本，干净对应"下发→稳定后观测"。
- **单一时钟**：`t0 = time.monotonic()` 在开始录制时定；frame_times/actions6/ndi 全是
  `monotonic()-t0` 相对秒，共享同一原点 → `capture_to_npz --actions-has-timestamps --frame-times`
  直接能用（且因 action 与 frame 同索引同时刻，插值退化为精确配对，无串扰）。
- **相机/NDI 自由运行**做预览 + 最新值缓存；时钟到点只读缓存（不阻塞、不丢拍）。
  PNG 落盘交给 `SaveThread`（异步 imwrite，不卡 GUI）。
- **单通道 / 向后兼容**：`active_channel` 指定主通道；除生成 7 列 `actions6.csv` 外，另写
  旧版 3 列 `pressure.csv = t, p_active, 0`，旧 `capture_to_npz` 文档命令照用。
- 全部 recorder 逻辑跑在**一个线程**（构造它的线程，通常是 GUI 主线程）：cam/ndi 跨线程
  信号经 Qt queued 连接进来，时钟/单次定时器也在该线程触发 → 缓存读写无竞争。

输出（每个序列目录，对齐 capture_to_npz schema）
-------------------------------------------------
  cam0/00000.png ...      零填充帧（cv2 缺失则跳过，frame_times 同步少写）
  frame_times.txt         每帧一行 相对秒           (--frame-times)
  actions6.csv            t_sec, c0..c5             (--actions --actions-has-timestamps)
  ndi.csv                 t_sec, x,y,z,Rx,Ry,Rz,qw,qx,qy,qz,quality  (→ tip.npz → --ndi-tip)
  pressure.csv            t_sec, p_active, 0        (旧 2 列兼容)
  meta.json               运行元信息
  summary.csv             (可选) 按帧对齐汇总
"""
from __future__ import annotations

import csv
import json
import math
import os
import queue
import time
from datetime import datetime

import numpy as np
from PyQt5.QtCore import QObject, QThread, QTimer, pyqtSignal, pyqtSlot

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

from valve_control import N_CHAN, P_MIN


def _now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


# ============================================================================
# PNG 落盘线程（把 ~10ms 的 imwrite 挪出 GUI 线程；哨兵 + 排空，不丢尾部帧）
# ============================================================================
class SaveThread(QThread):
    """异步 cv2.imwrite；cv2 缺失则全链路安全跳过（不入队，避免队列无限增长）。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._q: "queue.Queue" = queue.Queue()
        self._running = True

    def save(self, path: str, img: np.ndarray):
        if cv2 is None:
            return
        self._q.put((path, img))

    def run(self):
        if cv2 is None:
            return
        while True:
            try:
                item = self._q.get(timeout=0.2)
            except queue.Empty:
                if not self._running:
                    break
                continue
            if item is None:
                break
            path, img = item
            try:
                cv2.imwrite(path, img)
            except Exception as e:  # pragma: no cover
                print(f"[save] {path}: {e}")
        while True:  # 排空残余帧
            try:
                path, img = self._q.get_nowait()
            except queue.Empty:
                break
            try:
                cv2.imwrite(path, img)
            except Exception:
                pass

    def stop(self):
        self._running = False
        self._q.put(None)
        self.quit()
        self.wait(5000)


# ============================================================================
# ValveRecorder：同步核心（GUI 与 headless 共用）
# ============================================================================
class ValveRecorder(QObject):
    """采集核心。住 GUI 线程；camera/ndi 是跨线程生产者，controller 在同线程。

    动作门控：`_clock`（QTimer，period=action_interval_s）每拍下发一个 action，再用
    `singleShot(settle_s)` 延迟抓取缓存里的最新 frame/ndi 落盘。
    """

    # ---- 给 GUI 的信号 ----
    log = pyqtSignal(str)
    preview_frame = pyqtSignal(np.ndarray)
    pressure_status = pyqtSignal(list)                 # 6vec kPa（最新下发）
    ndi_status = pyqtSignal(list)                      # 11 维位姿（最新）
    recording_started = pyqtSignal(str)
    recording_status = pyqtSignal(int, float, list, float, float, float)  # frames, elapsed, action6, x, y, z
    recording_stopped = pyqtSignal(str, int)
    connection_changed = pyqtSignal(bool, str)
    group_connection_changed = pyqtSignal(int, bool)     # (group_id, connected) 透传给 GUI

    def __init__(self, cam, ndi, controller, parent=None):
        super().__init__(parent)
        self.cam = cam
        self.ndi = ndi
        self.controller = controller
        self.save_thread = SaveThread(self)
        self.save_thread.start()

        # 最新值缓存（GUI 线程独占读写 → 无锁）
        self._latest_frame = None
        self._frame_t = 0.0
        self._latest_ndi = [float("nan")] * 11
        self._ndi_t = 0.0

        # 录制状态
        self.recording = False
        self.t0 = 0.0
        self.seq_dir = ""
        self._cam_dir = ""
        self._f_frame = None
        self._f_act6 = None
        self._f_ndi = None
        self._f_pres = None
        self._act6_writer = None
        self._pres_writer = None
        self._meta = {}
        self._frame_idx = 0
        self._driver = None
        self._mode = "manual"
        self._manual_target = [P_MIN] * N_CHAN     # manual 模式每拍重发的目标
        self._active_channel = 0
        self._action_interval_s = 0.2
        self._settle_s = 0.19
        self._warned_no_cv2 = False
        self._warned_no_frame = False

        # 采集时钟（动作门控）
        self._clock = QTimer(self)
        self._clock.setSingleShot(False)
        self._clock.timeout.connect(self._on_tick)

        # 生产者 -> 本对象（跨线程 queued → 跑在本对象所在线程）
        cam.frame_ready.connect(self._on_cam)
        ndi.ndi_data.connect(self._on_ndi)
        if hasattr(controller, "action_logged"):
            controller.action_logged.connect(self._on_action)
        if hasattr(controller, "connection_changed"):
            controller.connection_changed.connect(self.connection_changed)
        if hasattr(controller, "group_connection_changed"):
            controller.group_connection_changed.connect(self.group_connection_changed)
        if hasattr(controller, "log"):
            controller.log.connect(self.log)

    # ---------------- 手动目标（manual 模式每拍重发）----------------
    def set_manual_target(self, pressures6):
        """GUI 改 spinbox 时推过来；manual 模式采集时钟每拍重发它。"""
        v = [float(x) for x in list(pressures6)[:N_CHAN]]
        if len(v) < N_CHAN:
            v += [P_MIN] * (N_CHAN - len(v))
        self._manual_target = v

    def update_ranges(self, lows6, highs6):
        """录制中实时改范围（random/sweep 下一拍生效）。未在录制 / 手动模式 → 无害 no-op。
        recorder 住 GUI 线程，`_on_tick` 也在该线程读 driver → 无竞争。"""
        if self._driver is None:
            return
        lo = [float(x) for x in list(lows6)[:N_CHAN]]
        hi = [float(x) for x in list(highs6)[:N_CHAN]]
        if len(lo) < N_CHAN:
            lo += [P_MIN] * (N_CHAN - len(lo))
        if len(hi) < N_CHAN:
            hi += [P_MIN] * (N_CHAN - len(hi))
        self._driver.set_ranges(lo, hi)

    # ---------------- 录制 ----------------
    def start_recording(self, seq_dir: str, mode: str, lows6, highs6,
                        action_interval_s: float, settle_s: float,
                        active_channel: int, note: str):
        if self.recording:
            self.log.emit("已在录制中。")
            return
        seq_dir = os.path.abspath(seq_dir)
        os.makedirs(seq_dir, exist_ok=True)
        self._cam_dir = os.path.join(seq_dir, "cam0")
        os.makedirs(self._cam_dir, exist_ok=True)
        self.seq_dir = seq_dir
        self.t0 = time.monotonic()
        self._frame_idx = 0
        self._mode = mode
        self._active_channel = int(active_channel)
        self._action_interval_s = max(0.02, float(action_interval_s))
        # settle 必须 < 间隔，留 5ms 给下一拍（防抓取跨入下一拍）
        self._settle_s = max(0.0, min(float(settle_s), self._action_interval_s - 0.005))
        self._warned_no_cv2 = False
        self._warned_no_frame = False

        lo = [float(x) for x in list(lows6)[:N_CHAN]]
        hi = [float(x) for x in list(highs6)[:N_CHAN]]
        if len(lo) < N_CHAN:
            lo += [P_MIN] * (N_CHAN - len(lo))
        if len(hi) < N_CHAN:
            hi += [P_MIN] * (N_CHAN - len(hi))

        self._f_frame = open(os.path.join(seq_dir, "frame_times.txt"), "w")
        self._f_act6 = open(os.path.join(seq_dir, "actions6.csv"), "w", newline="")
        self._act6_writer = csv.writer(self._f_act6)
        self._act6_writer.writerow(["t_sec", "c0", "c1", "c2", "c3", "c4", "c5"])   # 表头
        self._f_ndi = open(os.path.join(seq_dir, "ndi.csv"), "w", newline="")
        self._f_ndi.write("t_sec,x,y,z,Rx,Ry,Rz,qw,qx,qy,qz,quality\n")             # 表头
        self._f_pres = open(os.path.join(seq_dir, "pressure.csv"), "w", newline="")
        self._pres_writer = csv.writer(self._f_pres)
        self._pres_writer.writerow(["t_sec", "p_active", "reserved"])               # 表头

        self._meta = {
            "t0_monotonic": self.t0,
            "t0_wall": time.time(),
            "start_iso": _now_iso(),
            "mode": mode,
            "action_interval_s": self._action_interval_s,
            "settle_s": self._settle_s,
            "lo6": lo,
            "hi6": hi,
            "active_channel": self._active_channel,
            "note": note,
        }
        self.recording = True
        if cv2 is None:
            self.log.emit("⚠ cv2 未安装：无法存 PNG，图像链路停（actions6/ndi 仍记录）。pip install opencv-python")

        if mode in ("random", "sweep"):
            from valve_control import ValveDriver
            self._driver = ValveDriver(lo, hi, mode, parent=self)
            self._driver.reset()
            self.log.emit(f"自动驱动 [{mode}] 通道范围（kPa）："
                          + " | ".join(f"ch{i}:{lo[i]:.0f}-{hi[i]:.0f}" for i in range(N_CHAN)))
        else:
            self._driver = None
            self.log.emit("手动模式：每拍重发当前目标气压（GUI 改 spinbox 即时生效）。")

        self.recording_started.emit(seq_dir)
        self.log.emit(f"开始录制 -> {seq_dir}（动作间隔 {self._action_interval_s}s，"
                      f"稳定等待 {self._settle_s}s）")
        # 启动采集时钟；第一拍在 interval 后触发（给相机/NDI 缓存一点预热时间）
        self._clock.start(int(round(self._action_interval_s * 1000)))

    def _on_tick(self):
        """采集时钟：下发一拍 action，并安排 settle 后抓帧。"""
        if not self.recording:
            return
        action = self._driver.next_action() if self._driver is not None else list(self._manual_target)
        # 下发（emit action_logged → GUI 实时曲线；csv 在 _on_grab 用同一向量写）
        self.controller.set_pressures(action)
        # 安排 settle 后抓取（同一 action 向量 → (action_i, frame_i) 精确配对）
        QTimer.singleShot(int(round(self._settle_s * 1000)),
                           lambda a=list(action): self._on_grab(a))

    @pyqtSlot(list)
    def _on_grab(self, action):
        """settle 后：取缓存最新 frame/ndi，与 action 同索引落盘。"""
        if not self.recording:
            return
        t_grab = max(0.0, time.monotonic() - self.t0)
        idx = self._frame_idx

        # ---- 图像 ----
        frame = self._latest_frame
        if cv2 is not None and frame is not None:
            path = os.path.join(self._cam_dir, f"{idx:05d}.png")
            self.save_thread.save(path, frame)
            try:
                self._f_frame.write(f"{t_grab:.6f}\n")
                self._f_frame.flush()
            except Exception as e:
                self.log.emit(f"frame_times 写失败，安全停止: {e}")
                self.stop_recording()
                return
        else:
            if cv2 is None and not self._warned_no_cv2:
                self._warned_no_cv2 = True
            elif frame is None and not self._warned_no_frame:
                self._warned_no_frame = True
                self.log.emit("⚠ 相机尚未出帧：本拍跳过图像（actions6/ndi 仍记录）。")

        # ---- 动作 + NDI（同索引、同时刻 t_grab 落盘）----
        ndi = list(self._latest_ndi)
        try:
            self._act6_writer.writerow([f"{t_grab:.6f}"] + [f"{v:.4f}" for v in action])
            self._f_act6.flush()
            # ndi 失锁行写 "nan"（np.loadtxt 能解析；写空串会让 loadtxt 崩，下游 tip.npz/summary 会失败）
            ndi_cells = ["nan" if (isinstance(v, float) and math.isnan(v)) else f"{v:.6f}" for v in ndi]
            self._f_ndi.write(f"{t_grab:.6f}," + ",".join(ndi_cells) + "\n")
            self._f_ndi.flush()
            p_active = action[self._active_channel] if 0 <= self._active_channel < N_CHAN else 0.0
            self._pres_writer.writerow([f"{t_grab:.6f}", f"{p_active:.4f}", "0"])
            self._f_pres.flush()
        except Exception as e:
            self.log.emit(f"日志写失败，安全停止: {e}")
            self.stop_recording()
            return

        self._frame_idx += 1
        x, y, z = (ndi[0], ndi[1], ndi[2]) if len(ndi) >= 3 else (float("nan"),) * 3
        self.recording_status.emit(self._frame_idx, t_grab, list(action), float(x), float(y), float(z))

    def stop_recording(self):
        if not self.recording:
            return
        self.recording = False
        self._clock.stop()
        self._driver = None
        frames = self._frame_idx
        for f in (self._f_frame, self._f_act6, self._f_ndi, self._f_pres):
            try:
                if f is not None:
                    f.close()
            except Exception:
                pass
        self._f_frame = self._f_act6 = self._f_ndi = self._f_pres = None
        self._act6_writer = self._pres_writer = None
        self._meta.update(stop_iso=_now_iso(), frames=int(frames))
        try:
            with open(os.path.join(self.seq_dir, "meta.json"), "w") as fh:
                json.dump(self._meta, fh, indent=2, ensure_ascii=False)
        except Exception as e:
            self.log.emit(f"meta.json 写入失败: {e}")
        self.recording_stopped.emit(self.seq_dir, frames)
        self.log.emit(f"停止录制，共 {frames} 拍（帧）-> {self.seq_dir}")

    # ---------------- 生产者槽（更新缓存 + 推预览）----------------
    @pyqtSlot(np.ndarray, float)
    def _on_cam(self, img: np.ndarray, t_abs: float):
        self._latest_frame = img
        self._frame_t = float(t_abs)
        self.preview_frame.emit(img)

    @pyqtSlot(list, float)
    def _on_ndi(self, pose: list, t_abs: float):
        self._latest_ndi = list(pose) if len(pose) >= 11 else list(pose) + [float("nan")] * (11 - len(pose))
        self._ndi_t = float(t_abs)
        self.ndi_status.emit(self._latest_ndi)

    @pyqtSlot(list, float)
    def _on_action(self, action: list, t_abs: float):
        # 仅推 GUI 实时曲线；csv 在 _on_grab 写（避免双写）
        self.pressure_status.emit(list(action)[:N_CHAN])

    # ---------------- 生命周期（先停生产者/时钟，再 wait 落盘）----------------
    def shutdown(self):
        steps = (
            ("stop_recording", self.stop_recording),
            ("cam.stop", self.cam.stop),
            ("ndi", lambda: (self.ndi.stop(), self.ndi.wait(3000))),
            ("controller.zero+close", lambda: (self.controller.zero_all(), self.controller.close())),
            ("save_thread", lambda: (self.save_thread.stop(), self.save_thread.wait(5000))),
        )
        for label, fn in steps:
            try:
                fn()
            except Exception as e:
                print(f"[shutdown] {label}: {e}")


# ============================================================================
# 数值 CSV 读取（自动跳表头，兼容无表头旧文件）
# ============================================================================
def _load_num_csv(path, delimiter=","):
    """读数值 CSV；首行全 NaN 视为表头跳过。兼容带表头(新)与无表头(旧)两种。"""
    raw = np.atleast_2d(np.genfromtxt(path, delimiter=delimiter, dtype=float))
    while raw.shape[0] and np.isnan(raw[0]).all():
        raw = raw[1:]
    return raw


# ============================================================================
# 后处理：ndi.csv → tip.npz（喂 capture_to_npz --ndi-tip）
# ============================================================================
def build_ndi_tip_npz(seq_dir: str, out_path: str | None = None) -> str:
    """把 ndi.csv 的 xyz 按帧时刻插值成 tip.npz（字段 tip=(N,3)）。

    NaN 末端（失锁）会被 np.interp 用相邻有效值线性填补（端点外外推）。
    """
    ft_path = os.path.join(seq_dir, "frame_times.txt")
    ndi_path = os.path.join(seq_dir, "ndi.csv")
    if not os.path.isfile(ft_path) or not os.path.isfile(ndi_path):
        raise FileNotFoundError(f"需要 frame_times.txt 与 ndi.csv（{seq_dir}）")
    ft = np.loadtxt(ft_path)
    if ft.ndim == 0:
        ft = ft.reshape(1)
    raw = _load_num_csv(ndi_path)
    if raw.shape[1] < 4:
        raise ValueError("ndi.csv 至少需 4 列 (t,x,y,z)")
    t = raw[:, 0]
    tip = np.zeros((len(ft), 3), np.float32)
    for axis in range(3):
        col = raw[:, 1 + axis]
        mask = np.isfinite(col) & np.isfinite(t)
        if mask.sum() >= 1:
            tip[:, axis] = np.interp(ft, t[mask], col[mask])
        else:
            tip[:, axis] = 0.0
    if out_path is None:
        out_path = os.path.join(seq_dir, "tip.npz")
    np.savez(out_path, tip=tip)
    return out_path


# ============================================================================
# 汇总导出：按帧对齐成单表 CSV（气压 6 路 + NDI xyz + 图像名），便于人眼检查
# ============================================================================
def export_summary_csv(seq_dir: str, out_path: str | None = None) -> str:
    ft = np.loadtxt(os.path.join(seq_dir, "frame_times.txt"))
    if ft.ndim == 0:
        ft = ft.reshape(1)
    act = _load_num_csv(os.path.join(seq_dir, "actions6.csv"))
    ndi = _load_num_csv(os.path.join(seq_dir, "ndi.csv"))
    n = len(ft)
    if out_path is None:
        out_path = os.path.join(seq_dir, "summary.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame_idx", "t_s",
                    "p0", "p1", "p2", "p3", "p4", "p5",
                    "ndi_x", "ndi_y", "ndi_z", "quality", "image"])
        for i in range(n):
            row = [i, f"{ft[i]:.6f}"]
            row += [f"{act[i, 1 + c]:.4f}" if i < len(act) and act.shape[1] > 1 + c else ""
                    for c in range(6)]
            for axis in (0, 1, 2):
                row.append(f"{ndi[i, 1 + axis]:.4f}" if i < len(ndi) and ndi.shape[1] > 1 + axis else "")
            row.append(f"{ndi[i, 10]:.4f}" if i < len(ndi) and ndi.shape[1] > 10 else "")
            row.append(f"cam0/{i:05d}.png")
            w.writerow(row)
    return out_path
