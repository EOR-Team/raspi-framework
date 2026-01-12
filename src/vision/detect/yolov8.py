# yolov8.py
# YoloV8 (EdgeTPU.tflite) 模型 检测器
#
# @author n1ghts4kura
# @date 26-1-4
#

import numpy as np

from src import config
from src import logger
from src.typing import DetectionResult
from src.vision.detect.detector import Detector


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


class YoloV8Detector(Detector):
    """
    YoloV8 检测器类 (edgetpu tflite特化版)

    Usage:
        detector = YoloV8Detector(model_name="aimbot/yolov8n.26.1.2_fullint8_edgetpu.tflite")
        results = detector.invoke(image)
    """

    def __init__(self, model_name):
        super().__init__(model_name)


    def _resolve_outputs(self) -> list[DetectionResult]:
        """
        解析模型输出

        Returns:
            list[DetectionResult]: 检测结果列表
        """

        detections: list[DetectionResult] = []

        # 取出量化的输出 并反量化
        outputs = []
        for detail in self.output_details:
            raw = self.interpreter.get_tensor(detail['index'])
            outputs.append(dequantize(raw, detail))
        
        # 解析输出
        # 兼容单输出/多输出，这里默认用第一个输出作为 YOLO 预测张量
        preds = outputs[0]

        # 形状可能是 (1, anchors, channels) 或 (1, channels, anchors)
        pred = preds[0]
        if pred.shape[0] < pred.shape[1]:
            pred = pred.transpose(1, 0)

        boxes_xywh = pred[:, :4]
        class_scores = pred[:, 4:]

        if class_scores.size == 0:
            logger.error("模型输出没有类别分数，无法解析")
            raise RuntimeError("模型输出没有类别分数，无法解析")

        best_scores = class_scores.max(axis=1)
        best_cls = class_scores.argmax(axis=1)

        conf_mask = best_scores >= config.AIMBOT_CONF_THRESHOLD
        boxes_xywh = boxes_xywh[conf_mask]
        best_scores = best_scores[conf_mask]
        best_cls = best_cls[conf_mask]

        if boxes_xywh.size == 0:
            detections = []

        boxes_xyxy = xywh_to_xyxy(boxes_xywh, self.input_width, self.input_height)

        # 按类别独立做 NMS，可保留多目标多类别
        for cls_id in np.unique(best_cls):
            cls_mask = best_cls == cls_id
            cls_boxes = boxes_xyxy[cls_mask]
            cls_scores = best_scores[cls_mask]
            keep = nms(cls_boxes, cls_scores, config.AIMBOT_NMS_THRESHOLD)
            for idx in keep:
                detections.append( DetectionResult(
                    idx = cls_id,
                    conf = cls_scores[idx],
                    xyxyn = cls_boxes[idx].tolist()
                ) )

        # 统一按置信度排序
        detections.sort(key=lambda x: x.conf, reverse=True)
        return detections
    

