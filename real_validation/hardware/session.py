"""相机、阀和 NDI 的统一生命周期。

驱动仍是从 ``real_capture`` 复制进本目录的自包含实现；本模块只统一配置、状态、
Mock/Real 选择和安全关闭，不重新实现硬件协议。
"""

from __future__ import annotations

import time

from PyQt5.QtCore import QObject, pyqtSignal

from .profile import BackendMode, DeviceState, HardwareProfile


class HardwareSessionError(RuntimeError):
    pass


class HardwareSession(QObject):
    device_state_changed = pyqtSignal(str, str, str)  # device, DeviceState.value, message
    camera_frame = pyqtSignal(int, object, float)
    ndi_data = pyqtSignal(list, float)
    valve_command = pyqtSignal(list, float)
    log = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.profile = HardwareProfile.all_mock()
        self.cameras = []
        self.ndi_thread = None
        self.valve_controller = None
        self.states = {
            "camera": DeviceState.OFF,
            "valve": DeviceState.OFF,
            "ndi": DeviceState.OFF,
        }
        self.messages = {key: "未连接" for key in self.states}

    def _set_state(self, device: str, state: DeviceState, message: str) -> None:
        self.states[device] = DeviceState(state)
        self.messages[device] = str(message)
        self.device_state_changed.emit(device, state.value, str(message))

    def apply_profile(self, profile: HardwareProfile) -> None:
        if self.any_running:
            raise HardwareSessionError("应用运行配置前必须先断开全部硬件")
        self.profile = profile
        for device, backend in (("camera", profile.camera_backend),
                                ("valve", profile.valve_backend),
                                ("ndi", profile.ndi_backend)):
            if backend == BackendMode.DISABLED:
                self._set_state(device, DeviceState.DISABLED, "已禁用")
            else:
                self._set_state(device, DeviceState.OFF,
                                f"{backend.value.upper()} · 未连接")

    @property
    def any_running(self) -> bool:
        return bool(self.cameras or self.ndi_thread is not None
                    or self.valve_controller is not None)

    def start_cameras(self) -> None:
        backend = self.profile.camera_backend
        if backend == BackendMode.DISABLED:
            raise HardwareSessionError("相机已禁用")
        if self.cameras:
            raise HardwareSessionError("相机已启动")
        from .camera import RealSenseCam
        count = self.profile.camera_count
        if backend == BackendMode.MOCK:
            serials = [None] * count
        else:
            serials = list(self.profile.camera_serials)
            if not serials:
                serials = RealSenseCam.list_devices()[:count]
            if len(serials) != count:
                self._set_state("camera", DeviceState.ERROR,
                                f"请求 {count} 台，只发现 {len(serials)} 台")
                raise HardwareSessionError(self.messages["camera"])
            if len(set(serials)) != len(serials):
                raise HardwareSessionError("RealSense serial 重复")
        self._set_state("camera", DeviceState.CONNECTING,
                        f"启动 {backend.value.upper()} ×{count}")
        try:
            for index, serial in enumerate(serials):
                camera = RealSenseCam(mock=(backend == BackendMode.MOCK), serial=serial)
                camera.frame_ready.connect(
                    lambda image, stamp, idx=index: self._on_camera_frame(idx, image, stamp))
                camera.error.connect(
                    lambda message, idx=index: self._on_camera_error(idx, message))
                self.cameras.append(camera)
            for camera in self.cameras:
                camera.start()
            self._set_state("camera", DeviceState.READY,
                            f"{backend.value.upper()} ×{count}")
        except Exception:
            self.stop_cameras()
            raise

    def _on_camera_frame(self, index: int, image, timestamp: float) -> None:
        self.camera_frame.emit(int(index), image, float(timestamp))

    def _on_camera_error(self, index: int, message: str) -> None:
        self._set_state("camera", DeviceState.ERROR,
                        f"cam{index}: {message}")
        self.log.emit(f"相机 cam{index} 错误: {message}")

    def stop_cameras(self) -> None:
        cameras, self.cameras = list(self.cameras), []
        for camera in cameras:
            try:
                camera.stop()
            except Exception as error:
                self.log.emit(f"停止相机失败: {error}")
        backend = self.profile.camera_backend
        state = DeviceState.DISABLED if backend == BackendMode.DISABLED else DeviceState.OFF
        self._set_state("camera", state, "已禁用" if state == DeviceState.DISABLED
                        else f"{backend.value.upper()} · 已断开")

    def prepare_valves(self) -> object:
        backend = self.profile.valve_backend
        if backend == BackendMode.DISABLED:
            raise HardwareSessionError("阀已禁用")
        if self.valve_controller is not None:
            return self.valve_controller
        from .valve import MockValveController, ValveController
        if backend == BackendMode.MOCK:
            controller = MockValveController()
        else:
            ports = {}
            if self.profile.group1_port.strip():
                ports[1] = self.profile.group1_port.strip()
            if self.profile.group2_port.strip():
                ports[2] = self.profile.group2_port.strip()
            controller = ValveController(ports, self.profile.baudrate,
                                         self.profile.slave_addr)
        controller.action_logged.connect(self.valve_command)
        controller.log.connect(self.log)
        self.valve_controller = controller
        self._set_state("valve", DeviceState.CONNECTING,
                        f"{backend.value.upper()} · 等待连接阀组")
        return controller

    def connect_prepared_valves(self, groups: tuple[int, ...]) -> dict:
        controller = self.prepare_valves()
        from .valve import connect_valve_groups
        results = connect_valve_groups(controller, groups=groups)
        failed = {gid: message for gid, (ok, message) in results.items() if not ok}
        if failed:
            self._set_state("valve", DeviceState.ERROR,
                            f"连接失败: {failed}")
        else:
            self._set_state("valve", DeviceState.READY,
                            f"{self.profile.valve_backend.value.upper()} · 组{list(groups)}")
        return results

    def disconnect_valves(self, *, zero: bool = True) -> None:
        controller, self.valve_controller = self.valve_controller, None
        if controller is not None:
            try:
                if zero and controller.connected_groups:
                    controller.zero_all()
                    if hasattr(controller, "wait_idle"):
                        controller.wait_idle(1.0)
            except Exception as error:
                self.log.emit(f"阀归零失败: {error}")
            try:
                controller.close()
            except Exception as error:
                self.log.emit(f"关闭阀失败: {error}")
        backend = self.profile.valve_backend
        state = DeviceState.DISABLED if backend == BackendMode.DISABLED else DeviceState.OFF
        self._set_state("valve", state, "已禁用" if state == DeviceState.DISABLED
                        else f"{backend.value.upper()} · 已断开")

    def start_ndi(self) -> None:
        backend = self.profile.ndi_backend
        if backend == BackendMode.DISABLED:
            raise HardwareSessionError("NDI 已禁用")
        if self.ndi_thread is not None:
            raise HardwareSessionError("NDI 已启动")
        from .ndi import MockNdiThread, NdiThread
        thread = (MockNdiThread(ndi_count=self.profile.ndi_count)
                  if backend == BackendMode.MOCK else
                  NdiThread(self.profile.ndi_port, ndi_count=self.profile.ndi_count))
        thread.ndi_data.connect(self._on_ndi_data)
        if hasattr(thread, "error"):
            thread.error.connect(self._on_ndi_error)
        self.ndi_thread = thread
        self._set_state("ndi", DeviceState.CONNECTING,
                        f"启动 {backend.value.upper()} ×{self.profile.ndi_count}")
        thread.start()
        if backend == BackendMode.MOCK:
            self._set_state("ndi", DeviceState.READY,
                            f"MOCK ×{self.profile.ndi_count}")

    def _on_ndi_data(self, values: list, timestamp: float) -> None:
        if self.states["ndi"] != DeviceState.READY:
            self._set_state("ndi", DeviceState.READY,
                            f"{self.profile.ndi_backend.value.upper()} ×{self.profile.ndi_count}")
        self.ndi_data.emit(list(values), float(timestamp))

    def _on_ndi_error(self, message: str) -> None:
        self._set_state("ndi", DeviceState.ERROR, str(message))
        self.log.emit(f"NDI 错误: {message}")

    def stop_ndi(self) -> None:
        thread, self.ndi_thread = self.ndi_thread, None
        if thread is not None:
            try:
                thread.stop()
            except Exception as error:
                self.log.emit(f"停止 NDI 失败: {error}")
        backend = self.profile.ndi_backend
        state = DeviceState.DISABLED if backend == BackendMode.DISABLED else DeviceState.OFF
        self._set_state("ndi", state, "已禁用" if state == DeviceState.DISABLED
                        else f"{backend.value.upper()} · 已断开")

    def require_valves_ready(self, groups) -> None:
        if self.profile.valve_backend == BackendMode.DISABLED:
            raise HardwareSessionError("阀 backend 已禁用，不能执行")
        if self.states["valve"] != DeviceState.READY or self.valve_controller is None:
            raise HardwareSessionError("阀未 READY，禁止执行；不会自动回退 Mock")
        missing = set(int(value) for value in groups) - set(self.valve_controller.connected_groups)
        if missing:
            raise HardwareSessionError(f"必需阀组未连接: {sorted(missing)}")

    def create_transport(self, required_groups):
        self.require_valves_ready(required_groups)
        from ..execution.hardware_session import QtValveTransport
        return QtValveTransport(self.valve_controller)

    def snapshot(self) -> dict:
        return {
            "profile": self.profile.to_dict(),
            "states": {key: value.value for key, value in self.states.items()},
            "messages": dict(self.messages),
            "timestamp": time.time(),
        }

    def shutdown(self) -> None:
        self.stop_cameras()
        self.stop_ndi()
        self.disconnect_valves(zero=True)
