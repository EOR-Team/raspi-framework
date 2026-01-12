# config/camera.py
# 摄像头配置文件
#
# @author n1ghts4kura
# @date 26-01-12
#

# === 基本设置 ===
CAMERA_INDEX         =     0   # 摄像头索引 默认0
CAMERA_FOURCC        = "MJPG"  # 摄像头编码格式
CAMERA_AUTO_EXPOSURE =     1   # 自动曝光模式
CAMERA_EXPOSURE      =    64   # 曝光时间

# === 采集设置 ===
CAMERA_WIDTH_COL     =   640   # 摄像头分辨率宽度 (用于数据采集)
CAMERA_HEIGHT_COL    =   480   # 摄像头分辨率高度 (用于数据采集)
CAMERA_FPS_COL       =    60   # 摄像头帧率       (用于数据采集)

# === 推理设置 ===
CAMERA_WIDTH_INF     =   800   # 摄像头分辨率宽度 (用于推理)
CAMERA_HEIGHT_INF    =   600   # 摄像头分辨率高度 (用于推理)
CAMERA_FPS_INF       =    30   # 摄像头帧率       (用于推理)
