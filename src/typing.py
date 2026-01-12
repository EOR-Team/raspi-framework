# typing.py
# 定义
#
# @author n1ghts4kura
# @date 26-01-12
#

from dataclasses import dataclass


@dataclass
class DetectionResult:
    """
    检测结果
    """

    idx: int # 目标ID
    conf: float # 置信度
    xyxyn: tuple[float, float, float, float] # 归一化边界框 (x_min, y_min, x_max, y_max)

    def __str__(self):
        return f"DetectionResult(idx={self.idx}, conf={self.conf:.2f}, x1={self.xyxyn[0]:.2f}, y1={self.xyxyn[1]:.2f}, x2={self.xyxyn[2]:.2f}, y2={self.xyxyn[3]:.2f})"


