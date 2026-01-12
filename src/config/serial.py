# config/serial.py
# 串口配置文件
#
# @author n1ghts4kura
# @date 26-01-12
#

import serial as s

SERIAL_PORT          = "/dev/ttyUSB0"  # 串口设备路径 ttyUSB0为外置USB转串口设备
SERIAL_BAUDRATE      = 115200          # 串口波特率
SERIAL_TIMEOUT       = 1               # 串口超时时间（秒）
SERIAL_BYTESIZE      = s.EIGHTBITS     # 串口数据位
SERIAL_PARITY        = s.PARITY_NONE   # 串口校验位
SERIAL_STOPBITS      = s.STOPBITS_ONE  # 串口停止位
SERIAL_EOL           = "\n"            # 串口通信单行结束符
SERIAL_RX_DELAY      = 0.05            # 串口接收线程轮询延时（秒）
SERIAL_TX_DELAY      = 0.10            # 串口发送线程轮询延时（秒）
