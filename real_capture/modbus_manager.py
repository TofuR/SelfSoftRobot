try:
    import serial                       # pyserial（Modbus RTU 串口）
    import serial.tools.list_ports      # noqa: F401  (list_available_ports 用)
except Exception:                       # 未装时 mock / 仅相机+NDI 模式仍可用
    serial = None
import queue
import time
from PyQt5.QtCore import QObject, pyqtSignal, QThread
import struct


class ModbusRTU:
    """Modbus RTU协议实现

    硬件升级后控制组1、控制组2均采用 4-20mA 控制：
    4-20mA对应0-500kPa (寄存器值4000-20000)

    注：pressure_to_register_value_voltage / register_value_to_pressure_voltage
    为旧版 0-5V 模式的换算方法，升级后已不再使用，仅作历史保留。
    """

    @staticmethod
    def calculate_crc(data):
        """计算CRC16校验码"""
        crc = 0xFFFF
        for byte in data:
            crc ^= byte
            for _ in range(8):
                if crc & 0x0001:
                    crc >>= 1
                    crc ^= 0xA001
                else:
                    crc >>= 1
        return crc

    @staticmethod
    def write_single_register(slave_addr, reg_addr, value):
        """写单个保持寄存器 (功能码06H)"""
        # 构建数据帧
        data = bytearray([
            slave_addr,  # 从站地址
            0x06,  # 功能码
            (reg_addr >> 8) & 0xFF,  # 寄存器地址高字节
            reg_addr & 0xFF,  # 寄存器地址低字节
            (value >> 8) & 0xFF,  # 数据高字节
            value & 0xFF  # 数据低字节
        ])

        # 计算并添加CRC
        crc = ModbusRTU.calculate_crc(data)
        data.append(crc & 0xFF)  # CRC低字节
        data.append((crc >> 8) & 0xFF)  # CRC高字节

        return data

    @staticmethod
    def write_multiple_registers(slave_addr, start_addr, values):
        """写多个保持寄存器 (功能码10H)"""
        reg_count = len(values)
        byte_count = reg_count * 2

        # 构建数据帧
        data = bytearray([
            slave_addr,  # 从站地址
            0x10,  # 功能码
            (start_addr >> 8) & 0xFF,  # 起始寄存器地址高字节
            start_addr & 0xFF,  # 起始寄存器地址低字节
            (reg_count >> 8) & 0xFF,  # 寄存器数量高字节
            reg_count & 0xFF,  # 寄存器数量低字节
            byte_count  # 字节数
        ])

        # 添加寄存器值
        for value in values:
            data.append((value >> 8) & 0xFF)  # 高字节
            data.append(value & 0xFF)  # 低字节

        # 计算并添加CRC
        crc = ModbusRTU.calculate_crc(data)
        data.append(crc & 0xFF)  # CRC低字节
        data.append((crc >> 8) & 0xFF)  # CRC高字节

        return data

    @staticmethod
    def pressure_to_register_value_current(pressure_kpa):
        """将气压值(kPa)转换为寄存器值 - 4-20mA模式(控制组1)

        4-20mA电流对应0-500kPa气压
        寄存器值范围：
        - 4mA对应寄存器值 = 4000 (对应0kPa)
        - 20mA对应寄存器值 = 20000 (对应500kPa)

        线性关系：寄存器值 = 4000 + (pressure_kpa / 500) * (20000 - 4000)
                        = 4000 + pressure_kpa * 32
        """
        # 限制气压范围在0-500kPa
        pressure_kpa = max(0, min(500, pressure_kpa))

        # 4mA对应4000，20mA对应20000
        min_register = 4000  # 4mA对应的寄存器值
        max_register = 20000  # 20mA对应的寄存器值

        # 线性转换
        register_value = min_register + (pressure_kpa / 500.0) * (max_register - min_register)

        return int(register_value)

    @staticmethod
    def pressure_to_register_value_voltage(pressure_kpa):
        """将气压值(kPa)转换为寄存器值 - 0-5V模式(控制组2)

        0-5V电压对应0-500kPa气压
        寄存器值范围：
        - 0V对应寄存器值 = 0 (对应0kPa)
        - 5V对应寄存器值 = 5000 (对应500kPa)
        
        注：寄存器值采用固定3位小数，所以1V=1000

        线性关系：寄存器值 = (pressure_kpa / 500) * 5000
                        = pressure_kpa * 10
        """
        # 限制气压范围在0-500kPa
        pressure_kpa = max(0, min(500, pressure_kpa))

        # 0V对应0，5V对应5000
        min_register = 0  # 0V对应的寄存器值
        max_register = 5000  # 5V对应的寄存器值

        # 线性转换
        register_value = (pressure_kpa / 500.0) * max_register

        return int(register_value)

    @staticmethod
    def register_value_to_pressure_current(register_value):
        """将寄存器值转换为气压值(kPa) - 4-20mA模式"""
        min_register = 4000  # 4mA对应的寄存器值
        max_register = 20000  # 20mA对应的寄存器值

        # 限制寄存器值范围
        register_value = max(min_register, min(max_register, register_value))

        # 线性转换回气压值
        pressure_kpa = ((register_value - min_register) / (max_register - min_register)) * 500.0

        return pressure_kpa

    @staticmethod
    def register_value_to_pressure_voltage(register_value):
        """将寄存器值转换为气压值(kPa) - 0-5V模式"""
        min_register = 0  # 0V对应的寄存器值
        max_register = 5000  # 5V对应的寄存器值

        # 限制寄存器值范围
        register_value = max(min_register, min(max_register, register_value))

        # 线性转换回气压值
        pressure_kpa = (register_value / max_register) * 500.0

        return pressure_kpa


class ModbusThread(QThread):
    """Modbus通信线程"""
    data_sent = pyqtSignal(list, int, str, float)  # values, group, command_id, ack time
    error_occurred = pyqtSignal(str, int, str, float)  # status, group, command_id, time

    def __init__(self, serial_port, group_id):
        super().__init__()
        self.serial_port = serial_port
        self.group_id = group_id
        self.running = True
        # 有界队列，避免串口异常或过快采样时无限堆积命令。
        self.command_queue = queue.Queue(maxsize=64)
        self._busy = False

    def add_command(self, command_data):
        """添加命令到队列"""
        try:
            self.command_queue.put_nowait(command_data)
            return True
        except queue.Full:
            return False

    def _expected_response_len(self, function_code):
        # 0x06/0x10 正常响应固定8字节（地址+功能码+4字节+CRC）
        return 8 if function_code in (0x06, 0x10) else 8

    def _crc_ok(self, frame):
        if len(frame) < 4:
            return False
        recv_crc = frame[-2] | (frame[-1] << 8)
        calc_crc = ModbusRTU.calculate_crc(frame[:-2])
        return recv_crc == calc_crc

    def run(self):
        """线程运行"""
        while self.running:
            if not self.command_queue.empty() and self.serial_port.is_open:
                try:
                    command_data = self.command_queue.get_nowait()
                    command_id = str(command_data.get('command_id', ''))
                    self._busy = True

                    # 清理历史残留字节，避免上一帧干扰当前校验
                    self.serial_port.reset_input_buffer()

                    # 发送命令
                    sent_bytes = bytes(command_data['data'])
                    self.serial_port.write(sent_bytes)
                    self.serial_port.flush()
                    print(f"[组{self.group_id}] 发送({len(sent_bytes)}字节): {sent_bytes.hex(' ').upper()}")

                    # 读取固定长度响应，避免 in_waiting 的时序误判
                    expected_len = self._expected_response_len(command_data['function_code'])
                    response = self.serial_port.read(expected_len)

                    if len(response) != expected_len:
                        print(f"[组{self.group_id}] 无响应或长度不足: {len(response)}/{expected_len}")
                        self.error_occurred.emit("timeout", self.group_id, command_id, time.monotonic())
                        continue

                    print(f"[组{self.group_id}] 收到({len(response)}字节): {response.hex(' ').upper()}")

                    # Modbus异常响应: 功能码最高位置1
                    if response[1] & 0x80:
                        exc_code = response[2] if len(response) > 2 else -1
                        self.error_occurred.emit(f"exception_{exc_code}", self.group_id,
                                                 command_id, time.monotonic())
                        continue

                    # 验证响应
                    if self.validate_response(response, command_data):
                        self.data_sent.emit(command_data['values'], self.group_id,
                                            command_id, time.monotonic())
                    else:
                        self.error_occurred.emit("invalid_response", self.group_id,
                                                 command_id, time.monotonic())

                except Exception as e:
                    self.error_occurred.emit(f"serial_error:{e}", self.group_id,
                                             str(command_data.get('command_id', ''))
                                             if 'command_data' in locals() else "",
                                             time.monotonic())
                finally:
                    self._busy = False

            # 队列轮询间隔：原 50ms 把吞吐限制在 ~20 帧/s 并给每条命令增加最多
            # 50ms 下发延迟。压缩到 5ms 后，命令下发更跟手、采样写入更及时，
            # 对 CPU 占用影响可忽略（无命令时仅空转轻量轮询）。
            time.sleep(0.005)

    def validate_response(self, response, command_data):
        """验证响应数据"""
        if len(response) != self._expected_response_len(command_data['function_code']):
            return False

        # 检查CRC
        if not self._crc_ok(response):
            return False

        # 检查从站地址和功能码
        if response[0] != command_data['slave_addr'] or response[1] != command_data['function_code']:
            return False

        return True

    def stop(self):
        """停止线程"""
        self.running = False
        self.wait()

    def wait_idle(self, timeout_s=1.0):
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        while time.monotonic() < deadline:
            if self.command_queue.empty() and not self._busy:
                return True
            time.sleep(0.005)
        return self.command_queue.empty() and not self._busy


class ModbusManager(QObject):
    """Modbus管理类，管理多个控制组的串口连接"""

    # 定义信号
    pressure_data_sent = pyqtSignal(list, int)  # 兼容旧调用
    command_ack = pyqtSignal(str, int, float)  # command_id, group, ack time
    command_error = pyqtSignal(str, int, float, str)  # command_id, group, time, status
    connection_status_changed = pyqtSignal(str, bool)  # 连接状态变化

    def __init__(self):
        super().__init__()

        # 串口连接
        self.serial_ports = {}  # {group_id: serial.Serial}
        self.modbus_threads = {}  # {group_id: ModbusThread}

        # 连接参数
        self.connection_params = {}  # {group_id: {port, baudrate, slave_addr}}

        # 连接状态
        self.connection_status = {}  # {group_id: bool}

        # 寄存器地址映射 - 根据PDF文档（华控电子手册表2.1）
        # 000AH=第1路, 000BH=第2路, 000CH=第3路，两种设备地址相同
        # 硬件升级后两组均为 4-20mA：4mA=4000 / 20mA=20000
        self.register_addresses = {
            1: [0x000A, 0x000B, 0x000C],  # 控制组1: 第1-3路 (4-20mA)
            2: [0x000A, 0x000B, 0x000C]   # 控制组2: 第1-3路 (4-20mA)，地址与控制组1相同
        }

    def list_available_ports(self):
        """列出所有可用串口"""
        try:
            import serial.tools.list_ports
            ports = serial.tools.list_ports.comports()
            return [port.device for port in ports]
        except Exception as e:
            print(f"获取串口列表错误: {e}")
            return []

    def connect_group(self, group_id, port_name, baudrate=9600, slave_addr=1):
        """连接控制组

        Args:
            group_id: 控制组ID (1或2)
            port_name: 端口名
            baudrate: 波特率
            slave_addr: 从站地址
        Returns:
            (True, "") 成功，或 (False, 错误原因字符串) 失败
        """
        if group_id in self.serial_ports and self.serial_ports[group_id].is_open:
            self.disconnect_group(group_id)

        try:
            if serial is None:
                return False, "pyserial 未安装（pip install pyserial）"
            # 创建串口连接
            # 超时取值依据：9600bps 下 8 字节响应约 8.3ms 传输，叠加从站处理
            # 一般 <20ms。read() 收满预期字节即返回，因此 100ms 仅作"无响应"上限，
            # 既远大于正常往返、又显著压缩了丢帧时的最坏等待（原 300ms）。
            serial_port = serial.Serial(
                port=port_name,
                baudrate=baudrate,
                bytesize=8,
                parity='N',
                stopbits=1,
                timeout=0.1,
                write_timeout=0.1
            )
            serial_port.reset_input_buffer()
            serial_port.reset_output_buffer()

            print(f"[组{group_id}] 串口已打开: port={port_name}, baudrate={baudrate}, slave={slave_addr}")

            self.serial_ports[group_id] = serial_port
            self.connection_params[group_id] = {
                'port': port_name,
                'baudrate': baudrate,
                'slave_addr': slave_addr
            }

            # 创建并启动通信线程
            thread = ModbusThread(serial_port, group_id)
            thread.data_sent.connect(self.on_data_sent)
            thread.error_occurred.connect(self.on_error_occurred)
            thread.start()

            self.modbus_threads[group_id] = thread
            self.connection_status[group_id] = True

            self.connection_status_changed.emit(f"group_{group_id}", True)
            return True, ""

        except Exception as e:
            error_msg = str(e)
            print(f"控制组{group_id}串口连接错误: {error_msg}")
            self.connection_status[group_id] = False
            self.connection_status_changed.emit(f"group_{group_id}", False)
            return False, error_msg

    def disconnect_group(self, group_id):
        """断开控制组连接"""
        if group_id in self.modbus_threads:
            self.modbus_threads[group_id].stop()
            del self.modbus_threads[group_id]

        if group_id in self.serial_ports and self.serial_ports[group_id].is_open:
            self.serial_ports[group_id].close()
            del self.serial_ports[group_id]

        self.connection_status[group_id] = False
        self.connection_status_changed.emit(f"group_{group_id}", False)

    def set_pressure(self, group_id, channel_idx, pressure_kpa, command_id=None):
        """设置单通道气压

        Args:
            group_id: 控制组ID (1或2)
            channel_idx: 通道索引 (0-2)
            pressure_kpa: 气压值 (kPa) - 范围0-500kPa对应4-20mA
        """
        if group_id not in self.connection_status or not self.connection_status[group_id]:
            return False

        if group_id not in self.register_addresses or channel_idx >= len(self.register_addresses[group_id]):
            return False

        try:
            # 获取寄存器地址
            reg_addr = self.register_addresses[group_id][channel_idx]

            # 硬件升级后：控制组1、控制组2均为 4-20mA 控制，换算方法一致
            reg_value = ModbusRTU.pressure_to_register_value_current(pressure_kpa)

            # 获取从站地址
            slave_addr = self.connection_params[group_id]['slave_addr']

            # 构建Modbus命令
            command_data = ModbusRTU.write_single_register(slave_addr, reg_addr, reg_value)

            # 添加到线程队列
            if group_id in self.modbus_threads:
                return self.modbus_threads[group_id].add_command({
                    'data': command_data,
                    'slave_addr': slave_addr,
                    'function_code': 0x06,
                    'values': [pressure_kpa],
                    'channel': channel_idx,
                    'command_id': str(command_id or '')
                })

        except Exception as e:
            print(f"设置气压错误: {e}")

        return False

    def set_all_pressures(self, group_id, pressure_values, command_id=None):
        """设置控制组所有通道气压

        Args:
            group_id: 控制组ID (1或2)
            pressure_values: 气压值列表 [p1, p2, p3] (kPa) - 范围0-500kPa对应4-20mA
        """
        if group_id not in self.connection_status or not self.connection_status[group_id]:
            return False

        if group_id not in self.register_addresses:
            return False

        try:
            # 确保有3个值
            if len(pressure_values) < 3:
                pressure_values.extend([0.0] * (3 - len(pressure_values)))
            pressure_values = pressure_values[:3]

            # 获取起始寄存器地址
            start_addr = self.register_addresses[group_id][0]

            # 硬件升级后：控制组1、控制组2均为 4-20mA 控制，换算方法一致
            reg_values = [ModbusRTU.pressure_to_register_value_current(p) for p in pressure_values]

            # 获取从站地址
            slave_addr = self.connection_params[group_id]['slave_addr']

            # 构建Modbus命令
            command_data = ModbusRTU.write_multiple_registers(slave_addr, start_addr, reg_values)

            # 添加到线程队列
            if group_id in self.modbus_threads:
                return self.modbus_threads[group_id].add_command({
                    'data': command_data,
                    'slave_addr': slave_addr,
                    'function_code': 0x10,
                    'values': pressure_values,
                    'channel': -1,  # 表示所有通道
                    'command_id': str(command_id or '')
                })

        except Exception as e:
            print(f"设置所有气压错误: {e}")

        return False

    def on_data_sent(self, values, group_id, command_id, t_ack):
        """数据发送完成处理"""
        # 由于比例阀没有反馈，这里不需要再次发送pressure_data_sent信号
        # 因为在主界面中已经直接更新了实际值显示
        # 注释掉这行避免重复更新导致的显示错乱
        self.pressure_data_sent.emit(values, group_id)
        self.command_ack.emit(str(command_id), int(group_id), float(t_ack))

    def on_error_occurred(self, error_msg, group_id, command_id, t_error):
        """错误处理"""
        print(f"控制组{group_id}通信错误: {error_msg}")
        self.command_error.emit(str(command_id), int(group_id), float(t_error), str(error_msg))

    def is_group_connected(self, group_id):
        """检查控制组是否已连接"""
        return self.connection_status.get(group_id, False)

    def close_all(self):
        """关闭所有连接"""
        for group_id in list(self.serial_ports.keys()):
            self.disconnect_group(group_id)

    def wait_idle(self, timeout_s=1.0):
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        for thread in list(self.modbus_threads.values()):
            remaining = max(0.0, deadline - time.monotonic())
            thread.wait_idle(remaining)

