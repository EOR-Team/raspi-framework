# detector.py
# 检测器基类
#
# @author n1ghts4kura
# @date 26-01-04
#

# ==== **ATTENTION** ====
# 本检测器基类默认:
# - 输入图像大小 320px x 320px x 3colors
# =======================

from __future__ import annotations
import os
import cv2
import time
from typing import Any
from pycoral.utils import edgetpu
from pycoral.adapters import common

from src import logger
from src.typing import DetectionResult


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

        # frame = Image.fromarray(image.astype('uint8')).convert('RGB').resize((self.input_width, self.input_height))
        frame = cv2.resize(image, (self.input_width, self.input_height))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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

