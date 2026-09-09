"""相机、阀和 NDI 的统一生命周期。

驱动仍是从 ``real_capture`` 复制进本目录的自包含实现；本模块只统一配置、状态、
Mock/Real 选择和安全关闭，不重新实现硬件协议。
"""

from __future__ import annotations

import time
import threading
from collections import deque

from PyQt5.QtCore import QObject, Qt, pyqtSignal

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
        self.camera_epoch = 0
        self._frame_lock = threading.Lock()
        self._raw_frames = {}
        self._ndi_lock=threading.Lock();self._ndi_samples=deque(maxlen=60000);self._ndi_epoch=0
        self.camera_driver_active=None
        self.ndi_thread = None
        self.valve_controller = None
        self.last_applied6 = (0.0,) * 6
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

    def apply_disconnected_config(self, profile):
        """Change only settings belonging to disconnected devices/groups."""
        old=self.profile
        changed={key for key,value in old.to_dict().items() if value!=profile.to_dict()[key]}-{'name'}
        if not self.any_running:
            self.apply_profile(profile);return
        if self.cameras and changed & {'camera_backend','camera_count','camera_serials','camera_driver','camera_sources'}:
            raise HardwareSessionError('请先断开相机再更改其参数')
        c=self.valve_controller
        if c is not None:
            if changed & {'valve_backend','baudrate','slave_addr'}:
                raise HardwareSessionError('阀控制器已创建，请断开全部设备后更改 backend/串口协议')
            for gid,key in ((1,'group1_port'),(2,'group2_port')):
                if key in changed and gid in c.connected_groups:raise HardwareSessionError(f'请先断开组{gid}')
        if self.ndi_thread is not None and changed & {'ndi_backend','ndi_port','ndi_count'}:
            raise HardwareSessionError('请先断开 NDI')
        if c is not None and hasattr(c,'group_ports'):
            c.group_ports={1:profile.group1_port,2:profile.group2_port}
        self.profile=profile

    def start_cameras(self) -> None:
        backend = self.profile.camera_backend
        if backend == BackendMode.DISABLED:
            raise HardwareSessionError("相机已禁用")
        if self.cameras:
            raise HardwareSessionError("相机已启动")
        from .camera import RealSenseCam,OpenCVCam,camera_specs
        count=self.profile.camera_count
        driver,serials=camera_specs(self.profile);self.camera_driver_active=driver
        self.log.emit(f'相机使用 {driver}: {serials}；仅采集彩色图像')
        self._set_state("camera", DeviceState.CONNECTING,
                        f"启动 {backend.value.upper()} ×{count}")
        try:
            for index, serial in enumerate(serials):
                camera = OpenCVCam(source=serial) if driver=='opencv' else RealSenseCam(mock=(backend == BackendMode.MOCK), serial=serial)
                self.bind_frame_buffer(camera, index)
                camera.frame_ready.connect(
                    lambda image, stamp, idx=index, epoch=self.camera_epoch: self._on_camera_frame(idx, image, stamp, epoch))
                camera.error.connect(
                    lambda message, idx=index, epoch=self.camera_epoch: self._on_camera_error(idx, message, epoch))
                self.cameras.append(camera)
            for camera in self.cameras:
                camera.start()
            # Real devices become READY only after an actual frame arrives.
            if backend==BackendMode.MOCK:self._set_state("camera",DeviceState.READY,f'MOCK ×{count}')
        except Exception:
            self.stop_cameras()
            raise

    def _on_camera_frame(self, index: int, image, timestamp: float, epoch=None) -> None:
        if epoch is not None and epoch!=self.camera_epoch:return
        if self.states['camera']==DeviceState.CONNECTING:
            with self._frame_lock:complete=len(self._raw_frames)>=len(self.cameras)
            if complete:self._set_state('camera',DeviceState.READY,f'{self.camera_driver_active} ×{len(self.cameras)}')
        self.camera_frame.emit(int(index), image, float(timestamp))

    def bind_frame_buffer(self, camera, index: int) -> None:
        epoch = self.camera_epoch
        # Ingress runs in the capture emitter thread; no QWidget work here.
        camera.frame_ready.connect(
            lambda image, stamp, idx=index, generation=epoch:
                self._buffer_frame(idx, image, stamp, generation), Qt.DirectConnection)

    def _buffer_frame(self, index, image, timestamp, epoch) -> None:
        with self._frame_lock:
            if epoch == self.camera_epoch:
                self._raw_frames[int(index)] = (image.copy(), float(timestamp))

    def latest_camera_frame(self, index: int):
        with self._frame_lock:
            value = self._raw_frames.get(int(index))
            return None if value is None else (value[0].copy(), value[1])

    def _on_camera_error(self, index: int, message: str, epoch=None) -> None:
        if epoch is not None and epoch!=self.camera_epoch:return
        self._set_state("camera", DeviceState.ERROR,
                        f"cam{index}: {message}")
        self.log.emit(f"相机 cam{index} 错误: {message}")

    def stop_cameras(self) -> None:
        with self._frame_lock:
            self.camera_epoch += 1
            self._raw_frames.clear()
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
        controller.action_logged.connect(self._on_valve_command)
        controller.log.connect(self.log)
        self.valve_controller = controller
        self._set_state("valve", DeviceState.CONNECTING,
                        f"{backend.value.upper()} · 等待连接阀组")
        return controller

    def _on_valve_command(self, applied6, timestamp: float) -> None:
        values = tuple(float(value) for value in applied6)
        if len(values) != 6:
            self.log.emit(f"阀动作维度异常: {len(values)}")
            return
        self.last_applied6 = values
        self.valve_command.emit(list(values), float(timestamp))

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
        self._ndi_epoch+=1;epoch=self._ndi_epoch
        thread.ndi_data.connect(lambda values,stamp:self._buffer_ndi(values,stamp,epoch),Qt.DirectConnection)
        thread.ndi_data.connect(lambda values,stamp:self._on_ndi_data(values,stamp,epoch))
        if hasattr(thread, "error"):
            thread.error.connect(lambda message:self._on_ndi_error(message,epoch))
        self.ndi_thread = thread
        self._set_state("ndi", DeviceState.CONNECTING,
                        f"启动 {backend.value.upper()} ×{self.profile.ndi_count}")
        thread.start()
        if backend == BackendMode.MOCK:
            self._set_state("ndi", DeviceState.READY,
                            f"MOCK ×{self.profile.ndi_count}")

    def _buffer_ndi(self,values,timestamp,epoch):
        with self._ndi_lock:
            if epoch==self._ndi_epoch:self._ndi_samples.append((float(timestamp),list(values)))

    def evaluation_samples(self,started,ended):
        with self._ndi_lock:
            samples=[(stamp,list(values)) for stamp,values in self._ndi_samples if started<=stamp<=ended]
            oldest=self._ndi_samples[0][0] if self._ndi_samples else None
        return dict(backend=self.profile.ndi_backend.value,state=self.states['ndi'].value,
                    connected=self.ndi_thread is not None,samples=samples,
                    buffer_oldest=oldest,probe_count=self.profile.ndi_count)

    def camera_frames(self):
        with self._frame_lock:return {i:(frame.copy(),stamp) for i,(frame,stamp) in self._raw_frames.items()}

    def _on_ndi_data(self, values: list, timestamp: float, epoch=None) -> None:
        if epoch is not None and epoch!=self._ndi_epoch:return
        if self.states["ndi"] != DeviceState.READY:
            self._set_state("ndi", DeviceState.READY,
                            f"{self.profile.ndi_backend.value.upper()} ×{self.profile.ndi_count}")
        self.ndi_data.emit(list(values), float(timestamp))

    def _on_ndi_error(self, message: str, epoch=None) -> None:
        if epoch is not None and epoch!=self._ndi_epoch:return
        self._set_state("ndi", DeviceState.ERROR, str(message))
        self.log.emit(f"NDI 错误: {message}")

    def stop_ndi(self) -> None:
        self._ndi_epoch+=1
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
            "camera_driver_active": self.camera_driver_active,
            "states": {key: value.value for key, value in self.states.items()},
            "messages": dict(self.messages),
            "last_applied6": list(self.last_applied6),
            "timestamp": time.time(),
        }

    def shutdown(self) -> None:
        self.stop_cameras()
        self.stop_ndi()
        self.disconnect_valves(zero=True)
