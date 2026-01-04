# yolov8.py
# YoloV8 (EdgeTPU.tflite) 模型 检测器
#
# @author n1ghts4kura
# @date 26-1-4
#

from src import config
from src.vision.detect.detector import *
from src.vision.detect.detector import DetectionResult

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
    

