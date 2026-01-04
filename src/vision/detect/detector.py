# detector.py
# 检测器基类
#
# @author n1ghts4kura
# @date 26-1-4
#

# ==== **ATTENTION** ====
# 本检测器基类默认:
# - 输入图像大小 320px x 320px x 3colors
# =======================

from __future__ import annotations
from typing import Any
from dataclasses import dataclass
from pycoral.utils import edgetpu
from pycoral.adapters import common
from PIL import Image
import numpy as np
import os
import cv2
import time

from src import logger
from src import config


@dataclass
class DetectionResult:
    """
    检测结果
    """

    idx: int # 目标ID
    conf: float # 置信度
    xyxyn: tuple[float, float, float, float] # 归一化边界框 (x_min, y_min, x_max, y_max)


def dequantize(output: np.ndarray, detail: dict) -> np.ndarray:
    """
    反量化
    将量化后的INT8输出转换回浮点数表示

    Args:
        output (np.ndarray): 量化后的输出数组
        detail (dict): 输出张量的详细信息，包含量化参数
    Returns:
        np.ndarray: 反量化后的浮点数数组
    """

    scale, zero_point = detail.get('quantization', (0.0, 0))

    if scale == 0:
        return output.astype(np.float32)
    else:
        return ( output.astype(np.float32) - zero_point ) * scale


def xywh_to_xyxy(xywh: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
    """
    将边界框从 (x_center, y_center, width, height) 格式转换为 (x_min, y_min, x_max, y_max) 格式

    Args:
        xywh (np.ndarray): 边界框数组，形状为 (N, 4)
        img_w (int): 图像宽度
        img_h (int): 图像高度
    Returns:
        np.ndarray: 转换后的边界框数组，形状为 (N, 4)
    """

    x, y, w, h = np.split(xywh, 4, axis=-1)
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2
    # 约束边界框在图像范围内
    x1 = np.clip(x1, 0, img_w - 1)
    y1 = np.clip(y1, 0, img_h - 1)
    x2 = np.clip(x2, 0, img_w - 1)
    y2 = np.clip(y2, 0, img_h - 1)
    return np.concatenate([x1, y1, x2, y2], axis=1)


def iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """
    计算边界框的交并比 (IoU)

    Args:
        box (np.ndarray): 单个边界框，形状为 (4,)
        boxes (np.ndarray): 多个边界框，形状为 (N, 4)
    Returns:
        np.ndarray: IoU 数组，形状为 (N,)
    """

    # 计算交集坐标
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    # 计算交集面积
    inter_area = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

    # 计算各自面积
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    # 计算并集面积
    union_area = box_area + boxes_area - inter_area

    return np.divide(inter_area, union_area + 1e-6)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    """
    非极大值抑制 (NMS)
    用于去除重叠的边界框

    Args:
        boxes (np.ndarray): 边界框数组，形状为 (N, 4)
        scores (np.ndarray): 置信度数组，形状为 (N,)
        iou_threshold (float): IoU 阈值
    Returns:
        list[int]: 保留的边界框索引列表
    """

    if boxes.size == 0:
        return []

    idxs = scores.argsort()[::-1] # 按置信度降序排序索引
    keep = []

    while idxs.size > 0:
        i = int(idxs[0]) # 选择当前最高置信度的边界框
        keep.append(i) # 保留该边界框
        if idxs.size == 1:
            break # 仅剩一个边界框，结束循环

        rest = idxs[1:] # 剩余边界框索引
        ious = iou(boxes[i], boxes[rest]) # 计算与剩余边界框的IoU
        idxs = rest[ious < iou_threshold] # 保留IoU小于阈值的边界框索引

    return keep


class Detector:
    """
    检测器基类
    """

    def __init__(
        self,
        model_name: str,

    ) -> None:
        self.model_name = model_name
        self.interpreter: Any = None

        self._last_inf_time = 0.0
        self.input_width = 0
        self.input_height = 0
        self.output_details: Any = None
    

    def initialize(self) -> bool:
        """
        初始化检测器

        Returns:
            bool: 初始化是否成功
        """

        try:

            model_path = os.path.join("model", self.model_name)
            self.interpreter = edgetpu.make_interpreter(model_path)
            self.interpreter.allocate_tensors()

            self.input_width, self.input_height = common.input_size(self.interpreter)
            self.output_details = self.interpreter.get_output_details()

            logger.info(f"检测器模型 '{self.model_name}' 初始化成功")
            logger.info(f"输入形状: {(self.input_width, self.input_height)}")
            logger.info(f"输出: { [d['shape'] for d in self.output_details] }")

            return True
        
        except Exception as e:
            logger.error(f"检测器模型 '{self.model_name}' 初始化失败: {e}")
            return False
    

    def _inference(self, image: cv2.typing.MatLike) -> None:
        """
        运行推理

        Args:
            image (cv2.typing.MatLike): 输入图像
        """

        t0 = time.perf_counter()

        frame = Image.fromarray(image.astype('uint8')).convert('RGB').resize((self.input_width, self.input_height))
        common.set_input(self.interpreter, frame)
        self.interpreter.invoke()

        t1 = time.perf_counter()

        self._last_inf_time = (t1 - t0) * 1000.0 # 转换为毫秒
    

    def _resolve_outputs(self) -> list[DetectionResult]:
        """
        解析模型输出
        该方法需在子类中实现

        Returns:
            list[DetectionResult]: 检测结果列表
        """

        raise NotImplementedError("子类必须实现 '_resolve_outputs' 方法")
    

    def invoke(self, image: cv2.typing.MatLike) -> tuple[list[DetectionResult], float]:
        """
        执行检测

        Args:
            image (cv2.typing.MatLike): 输入图像
        Returns:
            tuple[list[DetectionResult], float]: 检测结果列表和推理时间（毫秒）
        """

        self._inference(image)
        return self._resolve_outputs(), self._last_inf_time
    
    @property
    def last_inference_time(self) -> float:
        """
        获取上次推理时间（毫秒）

        Returns:
            float: 上次推理时间（毫秒）
        """

        return self._last_inf_time

