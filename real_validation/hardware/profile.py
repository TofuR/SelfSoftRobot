"""可移植验证工作台的显式硬件配置契约。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum


class BackendMode(str, Enum):
    DISABLED = "disabled"
    MOCK = "mock"
    REAL = "real"


class DeviceState(str, Enum):
    DISABLED = "disabled"
    OFF = "off"
    CONNECTING = "connecting"
    READY = "ready"
    ERROR = "error"


@dataclass(frozen=True)
class HardwareProfile:
    name: str = "all_mock"
    camera_backend: BackendMode = BackendMode.MOCK
    camera_count: int = 1
    camera_serials: tuple[str, ...] = ()
    valve_backend: BackendMode = BackendMode.MOCK
    group1_port: str = "COM3"
    group2_port: str = "COM46"
    baudrate: int = 9600
    slave_addr: int = 1
    ndi_backend: BackendMode = BackendMode.MOCK
    ndi_port: str = "COM9"
    ndi_count: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "camera_backend", BackendMode(self.camera_backend))
        object.__setattr__(self, "valve_backend", BackendMode(self.valve_backend))
        object.__setattr__(self, "ndi_backend", BackendMode(self.ndi_backend))
        serials = tuple(str(value).strip() for value in self.camera_serials
                        if str(value).strip())
        if len(set(serials)) != len(serials):
            raise ValueError("RealSense serial 不能重复")
        object.__setattr__(self, "camera_serials", serials)
        if not 1 <= int(self.camera_count) <= 8:
            raise ValueError("camera_count 必须位于 1..8")
        if not 1 <= int(self.ndi_count) <= 8:
            raise ValueError("ndi_count 必须位于 1..8")
        if not 1200 <= int(self.baudrate) <= 115200:
            raise ValueError("baudrate 超出支持范围")
        if not 1 <= int(self.slave_addr) <= 247:
            raise ValueError("slave_addr 必须位于 1..247")
        if self.camera_backend == BackendMode.REAL and serials \
                and len(serials) != int(self.camera_count):
            raise ValueError("显式 camera_serials 数量必须等于 camera_count")
        if self.valve_backend == BackendMode.REAL \
                and not (self.group1_port.strip() or self.group2_port.strip()):
            raise ValueError("真阀模式至少需要一个串口")
        if self.ndi_backend == BackendMode.REAL and not self.ndi_port.strip():
            raise ValueError("真 NDI 模式必须填写串口")

    @classmethod
    def all_mock(cls) -> "HardwareProfile":
        return cls(name="all_mock")

    @classmethod
    def real(cls, **overrides) -> "HardwareProfile":
        values = dict(name="real", camera_backend=BackendMode.REAL,
                      valve_backend=BackendMode.REAL, ndi_backend=BackendMode.REAL)
        values.update(overrides)
        return cls(**values)

    def to_dict(self) -> dict:
        value = asdict(self)
        for key in ("camera_backend", "valve_backend", "ndi_backend"):
            value[key] = getattr(self, key).value
        return value

    @classmethod
    def from_dict(cls, value: dict) -> "HardwareProfile":
        data = dict(value)
        if "camera_serials" in data:
            data["camera_serials"] = tuple(data["camera_serials"])
        return cls(**data)


def required_groups_for_channels(channel_map) -> tuple[int, ...]:
    channels = tuple(int(value) for value in channel_map)
    if len(set(channels)) != len(channels) or any(value < 0 or value >= 6 for value in channels):
        raise ValueError("channel_map 必须是 0..5 内不重复通道")
    groups = set()
    if any(value <= 2 for value in channels):
        groups.add(1)
    if any(value >= 3 for value in channels):
        groups.add(2)
    return tuple(sorted(groups))
