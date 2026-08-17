# recorder.py
"""六通道阀 + NDI + 相机 的同步采集核心（动作门控 / action-gated）。

设计要点
--------
- **动作门控采集**（核心）：动作每 `action_interval_s`（默认 0.2s）下发一次；下发后等
  `settle_s`（默认 0.19s）让软臂稳定，再到缓存里取**最新的一帧 + 多个 NDI 末端位姿**落盘。
  → 每拍产出一组 `(action_i, frame_i, ndi_i)` 同索引样本，干净对应"下发→稳定后观测"。
- **单一时钟**：`t0 = time.monotonic()` 在开始录制时定；frame_times/actions6/ndi 全是
  `monotonic()-t0` 相对秒，共享同一原点 → `capture_to_npz --actions-has-timestamps --frame-times`
  直接能用（且因 action 与 frame 同索引同时刻，插值退化为精确配对，无串扰）。
- **相机/NDI 自由运行**做预览 + 最新值缓存；时钟到点只读缓存（不阻塞、不丢拍）。
  PNG 落盘交给 `SaveThread`（异步 imwrite，不卡 GUI）。
- **单通道 / 六通道共用**：单通道只需把其他通道范围设为 0；统一使用 `actions6.csv`，不再生成旧的
  `pressure.csv`。
- 全部 recorder 逻辑跑在**一个线程**（构造它的线程，通常是 GUI 主线程）：cam/ndi 跨线程
  信号经 Qt queued 连接进来，时钟/单次定时器也在该线程触发 → 缓存读写无竞争。

输出（每个序列目录，对齐 capture_to_npz schema）
-------------------------------------------------
  cam0/00000.png ...      每个相机一个目录（cv2 缺失则跳过，frame_times 同步少写）
  frame_times.txt         每帧一行 相对秒           (--frame-times)
  actions6.csv            t_sec, c0..c5             (--actions --actions-has-timestamps)
  ndi.csv                 t_sec + 每个探头 11 列位姿/质量
  commands.csv            命令时间、ACK、最终六通道命令、通信状态
  samples.csv             抓帧时间、总体/逐相机 frame_age、各 NDI age/quality
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
from PyQt5.QtCore import QObject, QThread, QTimer, Qt, pyqtSignal, pyqtSlot

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

from valve_control import (EQUALITY_TOLERANCE_KPA, N_CHAN, P_MIN,
                           ReplayDriver, ValveDriver,
                           apply_channel_equalities,
                           channel_equality_residuals,
                           normalize_channel_equalities)


def _now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


# ============================================================================
# PNG 落盘线程（把 ~10ms 的 imwrite 挪出 GUI 线程；哨兵 + 排空，不丢尾部帧）
# ============================================================================
class SaveThread(QThread):
    """异步 cv2.imwrite；cv2 缺失则全链路安全跳过（不入队，避免队列无限增长）。"""

    def __init__(self, parent=None, max_pending=8):
        super().__init__(parent)
        self._q: "queue.Queue" = queue.Queue(maxsize=max_pending)
        self._running = True

    def save(self, path: str, img: np.ndarray):
        return self.save_many([(path, img)])

    def save_many(self, items):
        """把同一拍的多路图像作为一个队列项入队，避免只写入部分视角。"""
        if cv2 is None:
            return False
        try:
            self._q.put_nowait(list(items))
            return True
        except queue.Full:
            return False

    def set_max_pending(self, max_pending):
        """调整批次队列容量；只在未录制时由相机数量变化触发。"""
        self._q.maxsize = max(1, int(max_pending))

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
            for path, img in item:
                try:
                    cv2.imwrite(path, img)
                except Exception as e:  # pragma: no cover
                    print(f"[save] {path}: {e}")
        while True:  # 排空残余帧
            try:
                items = self._q.get_nowait()
            except queue.Empty:
                break
            for path, img in items:
                try:
                    cv2.imwrite(path, img)
                except Exception:
                    pass

    def stop(self):
        self._running = False
        try:
            self._q.put_nowait(None)
        except queue.Full:
            pass
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
    preview_frames = pyqtSignal(list)                    # 多相机最新帧列表
    preview_frame_updated = pyqtSignal(int, np.ndarray)  # 单路更新，避免 GUI 重绘全部视角
    pressure_status = pyqtSignal(list)                 # 6vec kPa（最新下发）
    ndi_status = pyqtSignal(list)                      # 11 维位姿（最新）
    recording_started = pyqtSignal(str)
    recording_status = pyqtSignal(int, float, list, float, float, float)  # frames, elapsed, action6, x, y, z
    recording_stopped = pyqtSignal(str, int)
    connection_changed = pyqtSignal(bool, str)
    group_connection_changed = pyqtSignal(int, bool)     # (group_id, connected) 透传给 GUI

    def __init__(self, cam, ndi, controller, parent=None, cams=None):
        super().__init__(parent)
        self.cams = list(cams) if cams is not None else [cam]
        if not self.cams:
            raise ValueError("至少需要一个相机线程")
        self.cam = self.cams[0]  # 兼容旧调用方
        self.ndi = ndi
        self.controller = controller
        self.save_thread = SaveThread(self, max_pending=max(2, 8 // len(self.cams)))
        self.save_thread.start()

        # 最新值缓存（GUI 线程独占读写 → 无锁）
        self._latest_frame = None
        self._frame_t = 0.0
        self._latest_frames = [None] * len(self.cams)
        self._frame_ts = [0.0] * len(self.cams)
        self._camera_slots = []
        self._latest_ndi = [float("nan")] * 11
        self._ndi_t = 0.0

        # 录制状态
        self.recording = False
        self.t0 = 0.0
        self.seq_dir = ""
        self._cam_dir = ""          # cam0 兼容别的脚本
        self._cam_dirs = []
        self._f_frame = None
        self._f_act6 = None
        self._f_ndi = None
        self._f_cmd = None
        self._f_sample = None
        self._act6_writer = None
        self._cmd_writer = None
        self._sample_writer = None
        self._meta = {}
        self._frame_idx = 0
        self._driver = None
        self._replay = None
        self._replay_done = False
        self._pending_commands = {}
        self._ndi_count = 1
        self._max_frame_age = 0.5
        self._max_ndi_age = 0.5
        self._mode = "manual"
        self._manual_target = [P_MIN] * N_CHAN     # manual 模式每拍重发的目标
        self._active_channel = 0
        self._channel_equalities = ()
        self._action_interval_s = 0.2
        self._settle_s = 0.19
        self._warned_no_cv2 = False
        self._warned_no_frame = False

        # 采集时钟（动作门控）
        self._clock = QTimer(self)
        self._clock.setSingleShot(True)
        self._clock.timeout.connect(self._on_tick)

        # 生产者 -> 本对象（跨线程 queued → 跑在本对象所在线程）
        self._connect_cameras(self.cams)
        ndi.ndi_data.connect(self._on_ndi)
        if hasattr(controller, "action_logged"):
            controller.action_logged.connect(self._on_action)
        if hasattr(controller, "connection_changed"):
            controller.connection_changed.connect(self.connection_changed)
        if hasattr(controller, "group_connection_changed"):
            controller.group_connection_changed.connect(self.group_connection_changed)
        if hasattr(controller, "log"):
            controller.log.connect(self.log)
        if hasattr(controller, "communication_result"):
            controller.communication_result.connect(self._on_comm_result)

    def _connect_cameras(self, cams):
        self._camera_slots = []
        for index, camera in enumerate(cams):
            slot = lambda image, t, i=index: self._on_cam(i, image, t)
            camera.frame_ready.connect(slot, Qt.QueuedConnection)
            self._camera_slots.append(slot)

    def set_cameras(self, cams):
        """录制前替换相机线程集合；录制中禁止变更，避免帧索引失配。"""
        if self.recording:
            self.log.emit("⚠ 录制中不能修改相机数量，请先停止采集。")
            return False
        cams = list(cams)
        if not cams:
            raise ValueError("至少需要一个相机线程")
        for camera, slot in zip(self.cams, self._camera_slots):
            try:
                camera.frame_ready.disconnect(slot)
            except (TypeError, RuntimeError):
                pass
        self.cams = cams
        self.cam = cams[0]
        self._latest_frames = [None] * len(cams)
        self._frame_ts = [0.0] * len(cams)
        self._latest_frame = None
        self._frame_t = 0.0
        self.save_thread.set_max_pending(max(2, 8 // len(cams)))
        self._connect_cameras(cams)
        self.preview_frames.emit(list(self._latest_frames))
        return True

    def set_ndi_count(self, count: int):
        """设置 NDI 探头数；每个探头固定 11 列，缺失探头写 NaN。"""
        self._ndi_count = max(1, int(count))
        self._latest_ndi = [float("nan")] * (11 * self._ndi_count)

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
                        active_channel: int, note: str, rise_rates=None,
                        fall_rates=None, random_seed=None, pre_generate_steps=0,
                        replay_path=None, required_groups=None,
                        max_frame_age=0.5, max_ndi_age=0.5,
                        channel_equalities=()):
        if self.recording:
            self.log.emit("已在录制中。")
            return False
        try:
            equalities = normalize_channel_equalities(channel_equalities)
            def _six(values, fill):
                result = [float(x) for x in list(values or [])[:N_CHAN]]
                return result + [float(fill)] * (N_CHAN - len(result))

            rates_up = _six(rise_rates, 100.0)
            rates_down = _six(fall_rates, 100.0)
            lo_check = _six(lows6, P_MIN)
            hi_check = _six(highs6, P_MIN)
            for leader, follower in equalities:
                fields = (("min", lo_check), ("max", hi_check),
                          ("rise", rates_up), ("fall", rates_down))
                for name, values in fields:
                    if abs(values[leader] - values[follower]) > EQUALITY_TOLERANCE_KPA:
                        raise ValueError(
                            f"等值通道 ch{leader}/ch{follower} 的 {name} 必须相同")
            if hasattr(self.controller, "configure_channel_equalities"):
                self.controller.configure_channel_equalities(equalities)
            if hasattr(self.controller, "configure_safety"):
                self.controller.configure_safety(rates_up, rates_down)
        except (TypeError, ValueError) as error:
            self.log.emit(f"⚠ 通道等值约束无效：{error}")
            return False
        seq_dir = os.path.abspath(seq_dir)
        os.makedirs(seq_dir, exist_ok=True)
        self._cam_dirs = [os.path.join(seq_dir, f"cam{i}")
                          for i in range(len(self.cams))]
        for camera_dir in self._cam_dirs:
            os.makedirs(camera_dir, exist_ok=True)
        self._cam_dir = self._cam_dirs[0]
        self.seq_dir = seq_dir
        self.t0 = time.monotonic()
        self._frame_idx = 0
        self._mode = mode
        self._active_channel = int(active_channel)
        self._channel_equalities = equalities
        self._pending_commands.clear()
        self._replay_done = False
        self._max_frame_age = max(0.0, float(max_frame_age))
        self._max_ndi_age = max(0.0, float(max_ndi_age))
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
        self._replay = None
        if mode == "replay":
            if not replay_path:
                self.log.emit("⚠ replay 模式缺少 actions6.csv。")
                return
            try:
                self._replay = ReplayDriver(replay_path)
            except Exception as e:
                self.log.emit(f"⚠ replay 文件无效：{e}")
                return

        self._f_frame = open(os.path.join(seq_dir, "frame_times.txt"), "w")
        self._f_act6 = open(os.path.join(seq_dir, "actions6.csv"), "w", newline="")
        self._act6_writer = csv.writer(self._f_act6)
        self._act6_writer.writerow(["t_sec", "c0", "c1", "c2", "c3", "c4", "c5"])   # 表头
        self._f_ndi = open(os.path.join(seq_dir, "ndi.csv"), "w", newline="")
        ndi_fields = []
        for i in range(self._ndi_count):
            ndi_fields += [f"ndi{i}_{name}" for name in
                           ("x", "y", "z", "Rx", "Ry", "Rz", "qw", "qx", "qy", "qz", "quality")]
        self._f_ndi.write(",".join(["t_sec"] + ndi_fields) + "\n")
        self._f_cmd = open(os.path.join(seq_dir, "commands.csv"), "w", newline="")
        self._cmd_writer = csv.writer(self._f_cmd)
        self._cmd_writer.writerow(
            ["command_id", "t_command", "t_command_ack",
             *[f"requested{i}" for i in range(N_CHAN)],
             *[f"action_command{i}" for i in range(N_CHAN)],
             *[f"pair_residual{i}" for i in range(len(equalities))],
             "communication_status_g1", "communication_status_g2", "communication_status"])
        self._f_sample = open(os.path.join(seq_dir, "samples.csv"), "w", newline="")
        self._sample_writer = csv.writer(self._f_sample)
        self._sample_writer.writerow(
            ["frame_idx", "command_id", "t_grab", "frame_age"]
            + [f"frame_age{i}" for i in range(len(self.cams))]
            + [f"ndi{i}_age" for i in range(self._ndi_count)]
            + [f"ndi{i}_quality" for i in range(self._ndi_count)])

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
            "channel_equalities": [list(item) for item in equalities],
            "channel_equality_tolerance_kpa": EQUALITY_TOLERANCE_KPA,
            "camera_count": len(self.cams),
            "camera_serials": [getattr(camera, "serial", None) for camera in self.cams],
            "ndi_count": self._ndi_count,
            "random_seed": random_seed,
            "pre_generate_steps": int(pre_generate_steps),
            "replay_path": os.path.abspath(replay_path) if replay_path else "",
            "required_groups": sorted(int(g) for g in (required_groups or [])),
            "max_frame_age": self._max_frame_age,
            "max_ndi_age": self._max_ndi_age,
            "rise_rates6": list(rise_rates or []),
            "fall_rates6": list(fall_rates or []),
            "note": note,
        }
        if hasattr(self.controller, "set_required_groups"):
            self.controller.set_required_groups(required_groups)
        self.recording = True
        if cv2 is None:
            self.log.emit("⚠ cv2 未安装：无法进行有效采集。pip install opencv-python")

        if mode in ("random", "sweep"):
            self._driver = ValveDriver(lo, hi, mode, seed=random_seed, parent=self)
            self._driver.reset()
            if int(pre_generate_steps) > 0:
                self._driver.pre_generate(int(pre_generate_steps))
            self.log.emit(f"自动驱动 [{mode}] 通道范围（kPa）："
                          + " | ".join(f"ch{i}:{lo[i]:.0f}-{hi[i]:.0f}" for i in range(N_CHAN)))
        elif mode == "replay":
            self._driver = None
            self.log.emit(f"Replay：{replay_path}（{len(self._replay.actions)} 个动作）")
        else:
            self._driver = None
            self.log.emit("手动模式：每拍重发当前目标气压（GUI 改 spinbox 即时生效）。")

        self.recording_started.emit(seq_dir)
        self.log.emit(f"开始录制 -> {seq_dir}（动作间隔 {self._action_interval_s}s，"
                      f"稳定等待 {self._settle_s}s）")
        # 启动采集时钟；第一拍在 interval 后触发（给相机/NDI 缓存一点预热时间）
        first_delay = self._replay.next_delay(self._action_interval_s) if self._replay else self._action_interval_s
        self._clock.start(int(round(first_delay * 1000)))
        return True

    def _on_tick(self):
        """采集时钟：下发一拍 action，并安排 settle 后抓帧。"""
        if not self.recording:
            return
        if self._replay is not None:
            action = self._replay.next_action()
            if action is None:
                self._replay_done = True
                QTimer.singleShot(int(max(100, (self._settle_s + 0.3) * 1000)), self.stop_recording)
                return
        else:
            action = self._driver.next_action() if self._driver is not None else list(self._manual_target)
        action = apply_channel_equalities(action, self._channel_equalities)
        command_id = self.controller.allocate_command_id()
        self._pending_commands[command_id] = {
            "requested": list(action), "applied": list(action),
            "t_command": time.monotonic(), "acks": {1: None, 2: None},
            "statuses": {1: "pending", 2: "pending"},
            "required": set(getattr(self.controller, "_required_groups", set()) or set()),
        }
        try:
            result = self.controller.set_pressures(action, command_id=command_id)
        except Exception as error:
            self.log.emit(f"⚠ 动作下发违反等值/安全约束，停止采集：{error}")
            self.stop_recording()
            return
        if isinstance(result, tuple) and len(result) >= 2:
            self._pending_commands[command_id]["applied"] = list(result[1])
            if len(result) >= 3:
                self._pending_commands[command_id]["t_command"] = float(result[2])
        applied = list(self._pending_commands[command_id]["applied"])
        residuals = channel_equality_residuals(applied, self._channel_equalities)
        if any(value > EQUALITY_TOLERANCE_KPA for value in residuals):
            self.log.emit(f"⚠ applied6 等值残差 {residuals} 超限，停止采集。")
            self.stop_recording()
            return
        # 安排 settle 后抓取（同一 action 向量 → (action_i, frame_i) 精确配对）
        QTimer.singleShot(int(round(self._settle_s * 1000)),
                           lambda a=applied, cid=command_id: self._on_grab(a, cid))
        QTimer.singleShot(300, lambda cid=command_id: self._finalize_command(cid))
        if self._replay is not None:
            delay = self._replay.next_delay(self._action_interval_s)
            if delay is not None:
                self._clock.start(int(round(delay * 1000)))
        else:
            self._clock.start(int(round(self._action_interval_s * 1000)))

    @pyqtSlot(list, str)
    def _on_grab(self, action, command_id):
        """settle 后：取缓存最新 frame/ndi，与 action 同索引落盘。"""
        if not self.recording:
            return
        now_abs = time.monotonic()
        t_grab = max(0.0, now_abs - self.t0)
        idx = self._frame_idx

        # ---- 多相机图像：一拍要么完整写入所有视角，要么整拍丢弃 ----
        frames = list(self._latest_frames)
        frame_ages = [max(0.0, now_abs - t) if t > 0 else float("inf")
                      for t in self._frame_ts]
        bad_cameras = [i for i, (frame, age) in enumerate(zip(frames, frame_ages))
                       if frame is None or age > self._max_frame_age]
        if cv2 is None or bad_cameras:
            if not self._warned_no_frame:
                self._warned_no_frame = True
                self.log.emit(f"⚠ 相机帧无效/过期（camera={bad_cameras}，"
                              f"age={[round(a, 3) for a in frame_ages]}），"
                              "本拍不写入训练样本。")
            return
        image_items = [(os.path.join(self._cam_dirs[i], f"{idx:05d}.png"), frame)
                       for i, frame in enumerate(frames)]
        if not self.save_thread.save_many(image_items):
            self.log.emit("⚠ 图像写入队列已满，本拍所有视角一起丢弃以避免内存增长。")
            return
        frame_age = max(frame_ages)
        try:
            self._f_frame.write(f"{t_grab:.6f}\n")
            self._f_frame.flush()
        except Exception as e:
            self.log.emit(f"frame_times 写失败，安全停止: {e}")
            self.stop_recording()
            return
        # ---- 动作 + NDI（同索引、同时刻 t_grab 落盘）----
        ndi = list(self._latest_ndi)
        try:
            self._act6_writer.writerow([f"{t_grab:.6f}"] + [f"{v:.4f}" for v in action])
            self._f_act6.flush()
            ndi_cells = ["nan" if (isinstance(v, float) and math.isnan(v)) else f"{v:.6f}" for v in ndi]
            self._f_ndi.write(f"{t_grab:.6f}," + ",".join(ndi_cells) + "\n")
            self._f_ndi.flush()
            ndi_ages = []
            ndi_quality = []
            for i in range(self._ndi_count):
                t_ndi = self._ndi_t[i] if isinstance(self._ndi_t, list) and i < len(self._ndi_t) else self._ndi_t
                ndi_ages.append(max(0.0, now_abs - t_ndi) if t_ndi > 0 else float("inf"))
                q = ndi[i * 11 + 10] if len(ndi) >= (i + 1) * 11 else float("nan")
                ndi_quality.append(q)
            self._sample_writer.writerow(
                [idx, command_id, f"{t_grab:.6f}", f"{frame_age:.6f}"]
                + ["nan" if not math.isfinite(x) else f"{x:.6f}" for x in frame_ages]
                + ["nan" if not math.isfinite(x) else f"{x:.6f}" for x in ndi_ages]
                + ["nan" if not isinstance(x, (int, float)) or not math.isfinite(float(x)) else f"{float(x):.6f}"
                   for x in ndi_quality])
            self._f_sample.flush()
        except Exception as e:
            self.log.emit(f"日志写失败，安全停止: {e}")
            self.stop_recording()
            return

        self._frame_idx += 1
        x, y, z = (ndi[0], ndi[1], ndi[2]) if len(ndi) >= 3 else (float("nan"),) * 3
        self.recording_status.emit(self._frame_idx, t_grab, list(action), float(x), float(y), float(z))

    @pyqtSlot(str, int, bool, float, str)
    def _on_comm_result(self, command_id, group_id, ok, t_ack, status):
        record = self._pending_commands.get(str(command_id))
        if record is None:
            return
        gid = int(group_id)
        record["statuses"][gid] = str(status)
        if ok and str(status) == "ack":
            record["acks"][gid] = float(t_ack) - self.t0

    def _finalize_command(self, command_id):
        record = self._pending_commands.pop(str(command_id), None)
        if record is None or self._cmd_writer is None:
            return
        statuses = record["statuses"]
        required = record["required"]
        for gid in required:
            if statuses.get(gid) == "pending":
                statuses[gid] = "timeout"
        relevant = [statuses[g] for g in required] if required else list(statuses.values())
        if any(s in ("timeout", "not_connected", "queue_full") or s.startswith(("serial_error", "exception", "invalid"))
               for s in relevant):
            overall = next((s for s in relevant if s not in ("ack", "inactive")), "error")
        elif all(s in ("ack", "inactive") for s in relevant):
            overall = "ack"
        else:
            overall = "pending"
        ack_values = [v for v in record["acks"].values() if v is not None]
        t_ack = max(ack_values) if ack_values else float("nan")
        self._cmd_writer.writerow(
            [str(command_id), f"{record['t_command'] - self.t0:.6f}",
             "nan" if not math.isfinite(t_ack) else f"{t_ack:.6f}"]
            + [f"{float(v):.4f}" for v in record["requested"]]
            + [f"{float(v):.4f}" for v in record["applied"]]
            + [f"{value:.6f}" for value in channel_equality_residuals(
                record["applied"], self._channel_equalities)]
            + [statuses[1], statuses[2], overall])
        self._f_cmd.flush()

    def stop_recording(self):
        if not self.recording:
            return
        self.recording = False
        self._clock.stop()
        self._driver = None
        self._replay = None
        for command_id in list(self._pending_commands):
            self._finalize_command(command_id)
        frames = self._frame_idx
        for f in (self._f_frame, self._f_act6, self._f_ndi, self._f_cmd, self._f_sample):
            try:
                if f is not None:
                    f.close()
            except Exception:
                pass
        self._f_frame = self._f_act6 = self._f_ndi = self._f_cmd = self._f_sample = None
        self._act6_writer = self._cmd_writer = self._sample_writer = None
        self._meta.update(stop_iso=_now_iso(), frames=int(frames))
        try:
            with open(os.path.join(self.seq_dir, "meta.json"), "w") as fh:
                json.dump(self._meta, fh, indent=2, ensure_ascii=False)
        except Exception as e:
            self.log.emit(f"meta.json 写入失败: {e}")
        self.recording_stopped.emit(self.seq_dir, frames)
        self.log.emit(f"停止录制，共 {frames} 拍（帧）-> {self.seq_dir}")

    # ---------------- 生产者槽（更新缓存 + 推预览）----------------
    @pyqtSlot(int, np.ndarray, float)
    def _on_cam(self, index: int, img: np.ndarray, t_abs: float):
        if index < 0 or index >= len(self._latest_frames):
            return
        self._latest_frames[index] = img
        self._frame_ts[index] = float(t_abs)
        if index == 0:
            self._latest_frame = img
            self._frame_t = float(t_abs)
            self.preview_frame.emit(img)  # 兼容旧的单相机预览订阅者
        self.preview_frame_updated.emit(index, img)

    @pyqtSlot(list, float)
    def _on_ndi(self, pose: list, t_abs: float):
        expected = 11 * self._ndi_count
        self._latest_ndi = list(pose[:expected]) + [float("nan")] * max(0, expected - len(pose))
        self._ndi_t = float(t_abs)
        self.ndi_status.emit(self._latest_ndi)

    @pyqtSlot(list, float)
    def _on_action(self, action: list, t_abs: float):
        # 仅推 GUI 实时曲线；csv 在 _on_grab 写（避免双写）
        self.pressure_status.emit(list(action)[:N_CHAN])

    # ---------------- 生命周期（先停生产者/时钟，再 wait 落盘）----------------
    def shutdown(self):
        def zero_and_close():
            self.controller.zero_all()
            if hasattr(self.controller, "wait_idle"):
                self.controller.wait_idle(1.0)
            self.controller.close()

        steps = [("stop_recording", self.stop_recording)]
        steps.extend((f"cam{i}.stop", camera.stop) for i, camera in enumerate(self.cams))
        steps.extend([
            ("ndi", lambda: (self.ndi.stop(), self.ndi.wait(3000))),
            ("controller.zero+close", zero_and_close),
            ("save_thread", lambda: (self.save_thread.stop(), self.save_thread.wait(5000))),
        ])
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
    """把 ndi.csv 的多个 xyz 按帧时刻插值成 tip.npz。单探头保留 tip=(N,3)。"""
    ft_path = os.path.join(seq_dir, "frame_times.txt")
    ndi_path = os.path.join(seq_dir, "ndi.csv")
    if not os.path.isfile(ft_path) or not os.path.isfile(ndi_path):
        raise FileNotFoundError(f"需要 frame_times.txt 与 ndi.csv（{seq_dir}）")
    ft = np.loadtxt(ft_path)
    if ft.ndim == 0:
        ft = ft.reshape(1)
    raw = _load_num_csv(ndi_path)
    if raw.shape[1] < 4 or (raw.shape[1] - 1) % 11 != 0:
        raise ValueError("ndi.csv 列数必须为 1 + 11*n_ndi")
    t = raw[:, 0]
    n_ndi = (raw.shape[1] - 1) // 11
    tips = np.zeros((len(ft), n_ndi, 3), np.float32)
    for i in range(n_ndi):
        for axis in range(3):
            col = raw[:, 1 + i * 11 + axis]
            mask = np.isfinite(col) & np.isfinite(t)
            if mask.sum() >= 1:
                tips[:, i, axis] = np.interp(ft, t[mask], col[mask])
    if out_path is None:
        out_path = os.path.join(seq_dir, "tip.npz")
    if n_ndi == 1:
        np.savez(out_path, tip=tips[:, 0])
    else:
        np.savez(out_path, tips=tips)
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
    try:
        with open(os.path.join(seq_dir, "meta.json"), encoding="utf-8") as f:
            camera_count = int(json.load(f).get("camera_count", 1))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        camera_count = 1
        while os.path.isdir(os.path.join(seq_dir, f"cam{camera_count}")):
            camera_count += 1
    camera_count = max(1, camera_count)
    n = len(ft)
    if out_path is None:
        out_path = os.path.join(seq_dir, "summary.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        n_ndi = max(1, (ndi.shape[1] - 1) // 11)
        ndi_header = []
        for j in range(n_ndi):
            ndi_header += [f"ndi{j}_x", f"ndi{j}_y", f"ndi{j}_z", f"ndi{j}_quality"]
        image_header = (["image"] if camera_count == 1 else
                        [f"image{j}" for j in range(camera_count)])
        w.writerow(["frame_idx", "t_s", "p0", "p1", "p2", "p3", "p4", "p5"]
                   + ndi_header + image_header)
        for i in range(n):
            row = [i, f"{ft[i]:.6f}"]
            row += [f"{act[i, 1 + c]:.4f}" if i < len(act) and act.shape[1] > 1 + c else ""
                    for c in range(6)]
            for j in range(n_ndi):
                base = 1 + j * 11
                for col in (base, base + 1, base + 2, base + 10):
                    row.append(f"{ndi[i, col]:.4f}" if i < len(ndi) and ndi.shape[1] > col else "")
            if camera_count == 1:
                row.append(f"cam0/{i:05d}.png")
            else:
                row.extend(f"cam{j}/{i:05d}.png" for j in range(camera_count))
            w.writerow(row)
    return out_path
