"""实时流式推理工具：循环读取摄像头帧，经 PyCoral Edge TPU 推理并在终端实时刷新展示结果。

用法：
    python tests/run_stream_infer.py

说明：
- 依赖 config.AIMBOT_MODEL_PATH 指定的 EdgeTPU TFLite 模型；自动选择第一块 Edge TPU (usb:0)。
- 终端输出采用 ANSI 重绘，不刷屏；按 Ctrl+C 退出。
"""
import sys
import time
from collections import deque
from pathlib import Path
from typing import Deque, Tuple

import cv2
import numpy as np
from pycoral.utils.edgetpu import list_edge_tpus, make_interpreter

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config  # noqa: E402
from src.logger import get_logger  # noqa: E402
from src.vision.camera import InferCamera  # noqa: E402

logger = get_logger()

# 重绘间隔（秒），而不是每帧重绘，降低终端 I/O 开销
REPAINT_INTERVAL_SEC = 0.3
# 控制何时做后处理，减少在无关帧上的解析开销
POSTPROCESS_EVERY = 3


def select_device() -> str:
    edgetpus = list_edge_tpus()
    if not edgetpus:
        raise RuntimeError("未发现 Edge TPU 设备，请检查连接与权限")
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
    return device


def load_interpreter(model_path: Path, device: str):
    interp = make_interpreter(str(model_path), device=device)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    outs = interp.get_output_details()
    logger.info(
        "Input -> index=%s, shape=%s, dtype=%s, quant=%s",
        inp["index"], inp["shape"], inp["dtype"], inp["quantization"],
    )
    for od in outs:
        logger.info(
            "Output -> index=%s, shape=%s, dtype=%s, quant=%s",
            od.get("index"), od.get("shape"), od.get("dtype"), od.get("quantization"),
        )
    return interp, inp, outs


def preprocess(frame: np.ndarray, inp_shape: np.ndarray, inp_q: Tuple[float, int], dtype) -> np.ndarray:
    target_h, target_w = int(inp_shape[1]), int(inp_shape[2])  # NHWC 固定
    # 摄像头已配置与模型输入一致 (config.CAMERA_WIDTH_INF/HEIGHT_INF)，匹配时跳过 resize
    if frame.shape[0] == target_h and frame.shape[1] == target_w:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    else:
        resized = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    arr = rgb.astype(np.float32)
    scale, zero = inp_q
    if dtype != np.float32 and scale not in (0, None):
        arr = arr / scale + zero
        arr = np.clip(arr, 0, 255).astype(dtype)
    else:
        arr = arr.astype(np.float32)
    return np.expand_dims(arr, axis=0)


def postprocess(tensors: list[np.ndarray], orig_shape: Tuple[int, int]):
    if not tensors:
        return []
    first = tensors[0]
    if first.ndim != 3 or first.shape[1] != 6:
        return []
    preds = first.transpose(0, 2, 1)[0]  # (N,6) -> [x,y,w,h,score,cls_id]
    h, w = orig_shape
    results = []
    for row in preds:
        x_c, y_c, bw, bh, score, cls_id_val = row
        cls_id = int(round(cls_id_val))
        if score < 0.01:
            continue
        x_px = x_c * w if x_c <= 2 else x_c
        y_px = y_c * h if y_c <= 2 else y_c
        bw_px = bw * w if bw <= 2 else bw
        bh_px = bh * h if bh <= 2 else bh
        results.append((cls_id, float(score), (x_px, y_px, bw_px, bh_px)))
    return results


def redraw(status: str, detections: list[str]):
    # ANSI 清屏 + 光标归位
    sys.stdout.write("\033[2J\033[H")
    sys.stdout.write(status + "\n")
    for line in detections:
        sys.stdout.write(line + "\n")
    sys.stdout.flush()


def main() -> int:
    model_path = Path(config.AIMBOT_MODEL_PATH)
    device = select_device()
    interp, inp, outs = load_interpreter(model_path, device)

    cam = InferCamera()
    if not cam.open():
        logger.error("摄像头打开失败")
        return 1

    frame_count = 0
    # 窗口帧时间，用于更平滑/准确的短期 FPS，避免累计平均随时间漂移
    time_window: Deque[float] = deque(maxlen=90)
    # 记录上次重绘时间
    last_repaint = time.perf_counter()

    # 预取输出张量信息，避免循环内重复查询
    out0 = outs[0]
    out_index = out0["index"]
    out_scale, out_zero = out0["quantization"]

    try:
        while True:
            ret, frame = cam.read()
            if not ret or frame is None:
                redraw("未能从摄像头读取帧", [])
                time.sleep(0.05)
                continue

            tensor = preprocess(frame, inp["shape"], inp["quantization"], inp["dtype"])
            interp.set_tensor(inp["index"], tensor)
            t0 = time.perf_counter()
            interp.invoke()
            t1 = time.perf_counter()

            # 仅在需要展示时做后处理，减少解析与日志开销
            results = []
            need_post = (frame_count % POSTPROCESS_EVERY == 0)
            if need_post:
                out = interp.get_tensor(out_index)
                if out_scale not in (0, None) and out0["dtype"] != np.float32:
                    out = (out.astype(np.float32) - out_zero) * out_scale
                results = postprocess([out], frame.shape[:2])
            frame_count += 1
            elapsed = t1 - t0
            time_window.append(t1)

            fps = 0.0
            if len(time_window) >= 2:
                duration = time_window[-1] - time_window[0]
                if duration > 0:
                    fps = (len(time_window) - 1) / duration
            elif elapsed > 0:
                fps = 1.0 / elapsed

            now = time.perf_counter()
            if now - last_repaint >= REPAINT_INTERVAL_SEC:
                status = (
                    f"Edge TPU 实时推理 | FPS: {fps:.2f} | 当前帧耗时: {elapsed*1000:.2f} ms | 检测: {len(results)}"
                )
                lines = []
                for idx, (cls_id, score, (x, y, bw, bh)) in enumerate(results):
                    lines.append(
                        f"Det {idx}: cls={cls_id} score={score:.3f} xywh=({x:.1f},{y:.1f},{bw:.1f},{bh:.1f})"
                    )

                redraw(status, lines)
                last_repaint = now
    except KeyboardInterrupt:
        redraw("接收到退出信号，正在退出...", [])
    finally:
        try:
            cam.close()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
