# model_benchmark.py
# 模型跑分测试工具
#
# @author n1ghts4kura
# @date 26-1-3
#

import time
from collections import defaultdict

from src.vision import Camera
from src import logger, config
from src.vision.detect.yolov8 import YoloV8Detector

# 终端清屏序列（ANSI），用于“重绘”界面
CLEAR_SCREEN = "\033[2J\033[H"

# Init camera
cam = Camera(width=640, height=480)
logger.info(f"相机打开情况: {cam.open() and cam.test_opened()}")
logger.info(cam.get_actual_settings().__str__())

# Init detector
_detector = YoloV8Detector(config.AIMBOT_MODEL_PATH)
if not _detector.initialize():
    raise SystemExit("检测器初始化失败")

fps_last = time.time()
fps_smooth = 0.0

while True:
    try:
        ret, frame = cam.read()
        if not ret or frame is None:
            logger.error("无法从相机读取帧")
            continue

        detections, infer_ms = _detector.invoke(frame)

        # FPS 统计（平滑，逐帧更新）
        now = time.time()
        dt = now - fps_last
        if dt > 0:
            fps_inst = 1.0 / dt
            fps_smooth = fps_inst if fps_smooth == 0 else fps_smooth * 0.9 + fps_inst * 0.1
        fps_last = now

        # 按帧重绘输出，避免长日志；同时继续保留日志便于排查
        per_class_count = defaultdict(int)
        hud_lines = [
            f"当前FPS: {fps_smooth:.2f}",
            f"推理耗时: {infer_ms:.2f} ms",
            f"检测数量: {len(detections)}",
        ]
        for det in detections:
            cls_id = det.idx
            score = det.conf
            bbox = det.xyxyn
            per_class_count[cls_id] += 1
            bbox_str = "[" + ", ".join(f"{v:.3f}" for v in bbox) + "]"
            line = (
                f"检测到对象: ID={cls_id}, 序号={per_class_count[cls_id]}, "
                f"置信度={score:.2f}, 边界框={bbox_str}"
            )
            hud_lines.append(line)
            logger.info(line)

        # 清屏并打印 HUD（像游戏逐帧 Draw）
        print(CLEAR_SCREEN + "\n".join(hud_lines), end="", flush=True)
    except KeyboardInterrupt:
        logger.info("用户中断，退出程序")
        break
