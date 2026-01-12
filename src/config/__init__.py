# config/__init__.py
# 总配置文件
#
# @author n1ghts4kura
# @date 26-01-12
#

# === 调试配置 ===
DEBUG_MODE  = True  # 是否启用调试模式，启用后会打印更多日志信息
IMSHOW_ON   = False # 是否显示OpenCV2窗口 (请在有显示器连接时启用)
ANNOTATE_ON = False # 是否在显示的图像上绘制标注 (如检测框、关键点等)
# ================

from .aimbot import *
from .camera import *
from .serial import *
