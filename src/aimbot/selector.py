# selector.py
# 自瞄目标选择模块
#
# @author n1ghts4kura
# @date 25-12-6
# @version 1.0
#

import math
# from functools import wraps

from src.typing import DetectionResult


# # === 辅助函数 ===
# _weight_sum: float = 0

# def weight(weight: float):
#     """
#     选择器权重装饰器
#     用于给不同因素赋予不同权重
    
#     Args:
#         weight (float): 权重值
#     """

#     global _weight_sum
#     _weight_sum += weight

#     if _weight_sum > 1.0:
#         raise ValueError("选择器权重总和不能超过1.0")

#     def decorator(func):
#         @wraps(func)
#         def wrapper(*args, **kwargs):
#             return func(*args, **kwargs) * weight
#         return wrapper
#     return decorator

# === 选择器 v1.0 ===

# @weight(0.45)
def box_square_factor_v1(box: DetectionResult) -> float:
    """
    检测框面积 因子 v1 - 面积越大，离目标越近

    Args:
        box (Boxes): **单个** 目标边界框
    Returns:
        float: 检测框面积 因子，范围 [0.0, 1.0]
    """

    x1, y1, x2, y2 = box.xyxyn
    w = x2 - x1
    h = y2 - y1
    return float(w) * float(h)


# @weight(0.2)
# def box_center_factor_v1(box: DetectionResult) -> float:
#     """
#     检测框中心点 因子 v1 - 中心点越接近画面中心，离目标越近

#     Args:
#         box (Boxes): **单个** 目标边界框
#     Returns:
#         float: 检测框中心点 因子，范围 [0.0, 1.0]
#     """

#     x1, y1, x2, y2 = box.xyxyn
#     x = (x1 + x2) / 2 # 中心点x坐标
#     y = (y1 + y2) / 2 # 中心点y坐标
#     distance = math.sqrt(x ** 2 + y ** 2)
#     max_distance = 0.7071067811865476  # 对角线距离
#     return max(0.0, 1.0 - distance / max_distance) # => [0.0, 1.0]


# @weight(0.35)
def confidence_factor_v1(box: DetectionResult) -> float:
    """
    置信度 因子 v1 - 置信度越高，目标越可靠

    Args:
        box (Boxes): **单个** 目标边界框
    Returns:
        float: 置信度 因子，范围 [0.0, 1.0]
    """

    return float(box.conf)


def selector_v1(boxes: list[DetectionResult]) -> DetectionResult:
    """
    自瞄目标选择器 v1.0

    综合考虑检测框面积、中心点位置和置信度，选择最优目标。

    **注意**：确保boxes非空后再调用此函数。

    Args:
        boxes (list[Boxes]): 目标边界框列表
    Returns:
        Boxes: 选中的目标边界框
    """

    best_box = boxes[0]
    best_score = -1.0

    for box in boxes:
        score = (
            box_square_factor_v1(box) +
            # box_center_factor_v1(box) +
            confidence_factor_v1(box)
        )

        if score > best_score:
            best_score = score
            best_box = box

    return best_box


__all__ = [
    "selector_v1",
]