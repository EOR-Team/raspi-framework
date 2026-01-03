"""使用 PyCoral API 基于 Edge TPU 的自瞄模型跑分脚本。

使用说明：
- 依赖：确保已安装 pycoral（可用 assets/pycoral-2.0.0-cp39-cp39-linux_aarch64.whl 安装）；确保 Edge TPU USB 加速棒已连接。
- 模型：默认加载 config.AIMBOT_MODEL_PATH 指向的 TFLite EdgeTPU 模型。
- 运行：在虚拟环境中执行 `python tests/benchmark_coral.py --duration 60 --warmup 60`。
- 结果：日志输出 FPS、处理帧数与检测总数；如为 0 检测，检查模型输出格式是否与简单 YOLO 解析兼容。
"""
import argparse
import gc
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.edgetpu import list_edge_tpus

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.logger import get_logger, LogLevel  # noqa: E402
from src import config  # noqa: E402
from src.vision.camera import Camera, InferCamera  # noqa: E402


logger = get_logger()
logger.set_level(LogLevel.INFO)


@dataclass
class Detection:
    cls_id: int
    score: float
    xywh: tuple[float, float, float, float]


class PyCoralAimbotDetector:
    """使用 PyCoral 直接驱动 Edge TPU 的检测器，不依赖 Ultralytics。"""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.interpreter = None
        self.input_index = None
        self.input_shape = None
        self.input_dtype = None
        self.input_scale = None
        self.input_zero_point = None
        self.output_details = None

    def initialize(self) -> bool:
        try:
            # 优先选择首个可用 Edge TPU
            edgetpus = list_edge_tpus()
            if not edgetpus:
                raise RuntimeError("未发现 Edge TPU 设备，请检查连接与权限 (如 udev 规则)")

            if config.DEBUG_MODE:
                logger.debug(f"Edge TPU 枚举结果: {edgetpus}")
            first = edgetpus[0]
            if isinstance(first, dict):
                dev_type = first.get("type", "")
                device = f"{dev_type}:0" if dev_type else ":0"
            elif isinstance(first, (tuple, list)) and first:
                dev_type = first[0] if isinstance(first[0], str) else ""
                device = f"{dev_type}:0" if dev_type else ":0"
            else:
                device = ":0"

            logger.info(f"使用 Edge TPU 设备: {device} (原始枚举: {first})")

            self.interpreter = make_interpreter(self.model_path, device=device)
            self.interpreter.allocate_tensors()

            input_det = self.interpreter.get_input_details()[0]
            self.input_index = input_det["index"]
            self.input_shape = input_det["shape"]
            self.input_dtype = np.dtype(input_det["dtype"])
            self.input_scale, self.input_zero_point = input_det["quantization"]
            self.output_details = self.interpreter.get_output_details()

            if config.DEBUG_MODE:
                logger.debug(
                    "Input detail -> index: %s, shape: %s, dtype: %s, q: (%s, %s)",
                    self.input_index,
                    self.input_shape,
                    self.input_dtype,
                    self.input_scale,
                    self.input_zero_point,
                )
                for od in self.output_details or []:
                    logger.debug(
                        "Output detail -> index: %s, shape: %s, dtype: %s, q: %s",
                        od.get("index"),
                        od.get("shape"),
                        od.get("dtype"),
                        od.get("quantization"),
                    )
            return True
        except Exception as exc:  # pragma: no cover - 硬件相关
            logger.error(f"初始化 PyCoral 检测器失败: {exc}；已发现的 Edge TPU 列表: {list_edge_tpus()}")
            return False

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """BGR -> RGB，按模型输入分辨率缩放，并应用量化。"""

        if self.input_shape is None:
            raise RuntimeError("检测器未初始化")

        # 推断布局：TFLite 通常为 NHWC；少数导出为 NCHW
        if len(self.input_shape) != 4:
            raise RuntimeError(f"不支持的输入 shape: {self.input_shape}")

        if self.input_shape[3] == 3:  # NHWC
            target_h, target_w = int(self.input_shape[1]), int(self.input_shape[2])
            layout = "NHWC"
        elif self.input_shape[1] == 3:  # NCHW
            target_h, target_w = int(self.input_shape[2]), int(self.input_shape[3])
            layout = "NCHW"
        else:
            raise RuntimeError(f"无法识别输入通道位置: {self.input_shape}")

        resized = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        if layout == "NHWC":
            arr = rgb.astype(np.float32)
        else:
            arr = np.transpose(rgb, (2, 0, 1)).astype(np.float32)

        # 量化输入
        if self.input_scale not in (0, None) and self.input_dtype != np.float32:
            arr = arr / self.input_scale + self.input_zero_point
            arr = np.clip(arr, 0, 255).astype(self.input_dtype)
        else:
            arr = arr.astype(np.float32)

        batched = np.expand_dims(arr, axis=0)

        if config.DEBUG_MODE:
            logger.debug(
                "Preprocess -> frame: %s, target: (%s, %s), layout: %s, dtype: %s, batched shape: %s",
                frame.shape,
                target_h,
                target_w,
                layout,
                self.input_dtype,
                batched.shape,
            )

        return batched

    def _postprocess(self, tensors: list[np.ndarray], orig_shape: Tuple[int, int]) -> list[Detection]:
        """简单解析 YOLO 输出。如果格式无法识别，则返回空列表但不中断。"""

        # 模型输出固定：形状 (1, 6, N)，含 [x, y, w, h, score, cls_id]
        if not tensors:
            return []

        first = tensors[0]
        if first.ndim != 3 or first.shape[1] != 6:
            if config.DEBUG_MODE:
                logger.debug(f"Unexpected output shape: {first.shape}")
            return []

        preds = first.transpose(0, 2, 1)[0]  # -> (N, 6)

        h, w = orig_shape
        results: list[Detection] = []

        for row in preds:
            x_c, y_c, bw, bh, score, cls_id_val = row
            cls_id = int(round(cls_id_val))
            if score < 0.01:
                continue

            # 模型导出可能是归一化或像素值，尝试兼容两种情况
            x_px = x_c * w if x_c <= 2 else x_c
            y_px = y_c * h if y_c <= 2 else y_c
            bw_px = bw * w if bw <= 2 else bw
            bh_px = bh * h if bh <= 2 else bh

            results.append(Detection(cls_id=cls_id, score=score, xywh=(x_px, y_px, bw_px, bh_px)))

        return results

    def detect(self, frame: np.ndarray) -> list[Detection]:
        if self.interpreter is None or self.output_details is None:
            logger.error("检测器未初始化")
            return []

        input_tensor = self._preprocess(frame)
        self.interpreter.set_tensor(self.input_index, input_tensor)
        self.interpreter.invoke()

        tensors: list[np.ndarray] = []
        for det in self.output_details:
            out = self.interpreter.get_tensor(det["index"])
            scale, zero = det["quantization"]
            if scale not in (0, None) and det["dtype"] != np.float32:
                out = (out.astype(np.float32) - zero) * scale
            tensors.append(out)

        if config.DEBUG_MODE and tensors:
            logger.debug("Raw outputs -> count: %d, shapes: %s", len(tensors), [t.shape for t in tensors])

        h, w = frame.shape[:2]
        return self._postprocess(tensors, (h, w))


def warmup(detector: PyCoralAimbotDetector, camera: Camera, warmup_frames: int) -> None:
    completed = 0
    while completed < warmup_frames:
        ret, frame = camera.read()
        if not ret or frame is None:
            continue
        detector.detect(frame)
        completed += 1
    logger.info(f"预热完成: {completed} 帧")


def benchmark(detector: PyCoralAimbotDetector, camera: Camera, duration: float, max_frames: Optional[int]) -> dict:
    frames = 0
    detections = 0
    start = time.perf_counter()

    while True:
        now = time.perf_counter()
        if max_frames is not None and frames >= max_frames:
            break
        if now - start >= duration:
            break

        ret, frame = camera.read()
        if not ret or frame is None:
            continue

        results = detector.detect(frame)
        frames += 1
        detections += len(results)

    elapsed = time.perf_counter() - start
    fps = frames / elapsed if elapsed > 0 else 0.0
    return {
        "frames": frames,
        "elapsed": elapsed,
        "fps": fps,
        "detections": detections,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark Coral USB Accelerator with PyCoral")
    parser.add_argument("--duration", type=float, default=10.0, help="Benchmark duration in seconds")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup frames before timing")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional cap on processed frames")
    args = parser.parse_args()

    camera = InferCamera()
    detector: Optional[PyCoralAimbotDetector] = None

    try:
        if not camera.open():
            logger.error("摄像头打开失败")
            return 1

        actual = camera.get_actual_settings()
        if actual:
            logger.info(
                f"Camera opened: {actual['width']}x{actual['height']}@{actual['fps']}FPS fourcc={actual['fourcc']}"
            )

        detector = PyCoralAimbotDetector(model_path=config.AIMBOT_MODEL_PATH)
        if not detector.initialize():
            logger.error("PyCoral 检测器初始化失败")
            return 1

        logger.info("开始预热...")
        warmup(detector, camera, args.warmup)

        logger.info("开始正式跑分...")
        stats = benchmark(detector, camera, args.duration, args.max_frames)

        logger.info(
            f"Frames: {stats['frames']} | Elapsed: {stats['elapsed']:.3f}s | FPS: {stats['fps']:.2f} | "
            f"Total detections: {stats['detections']}"
        )

        return 0

    finally:
        try:
            camera.close()
        except Exception:
            pass

        gc.collect()


if __name__ == "__main__":
    raise SystemExit(main())
