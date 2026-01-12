# config/aimbot.py
# 自瞄技能配置文件
#
# @author n1ghts4kura
# @date 26-01-12
#

from simple_pid import PID


# === 模型选择 ===
AIMBOT_MODEL_PATH      = "aimbot/yolov8n.26.1.2_fullint8_edgetpu.tflite"  # 自瞄模型路径
AIMBOT_CONF_THRESHOLD  = 0.35 # 置信度阈值 低于该值的目标将被忽略
AIMBOT_NMS_THRESHOLD   = 0.45 # 非极大值抑制阈值 

# === PID参数配置 ===
OUTPUT_LIMIT = 100.0  # PID输出限制 单位 °/s

# == 水平轴 ==
_hor_kp = 120.0
_hor_ki =   4.0
_hor_kd =   2.0

AIMBOT_HOR_PID = PID(
    Kp=_hor_kp,
    Ki=_hor_ki,
    Kd=_hor_kd,
    setpoint=0.0,
    output_limits=(-OUTPUT_LIMIT, OUTPUT_LIMIT)
)

# == 垂直轴 ==
_ver_kp = 110.0
_ver_ki =   5.0
_ver_kd =   2.0

AIMBOT_VER_PID = PID(
    Kp=_ver_kp,
    Ki=_ver_ki,
    Kd=_ver_kd,
    setpoint=0.0,
    output_limits=(-OUTPUT_LIMIT, OUTPUT_LIMIT)
)

# == 输出判定 ==
"""
26-01-12 @n1ghts4kura:
    目前仍使用 画面比例为4:3的摄像头 而输入自瞄模型处理时会被压缩为1:1的正方形
    因此 死区阈值应该对应回4:3的比例 也就是 横向的死区阈值 应该更大一些（？）
    当然你不做这个操作也可以。
"""
AIMBOT_DEADZONE_HOR_DELTA = 0.025 # 水平轴 PID输出死区阈值 大小为其在整个屏幕占比
AIMBOT_DEADZONE_VER_DELTA = 0.025 # 垂直轴 PID输出死区阈值 大小为其在整个屏幕占比

AIMBOT_CROSSHAIR_X = 0.500 # 准星 X位置
AIMBOT_CROSSHAIR_Y = 0.605 # 准星 Y位置

# === 技能相关 ===
AIMBOT_ACTION_DELAY      = 1 / 60 # 技能一轮循环预期耗时 单位 秒
