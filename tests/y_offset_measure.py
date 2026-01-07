"""
y_offset 测量器（更鲁棒版）
临时脚本：测量新摄像头相对原摄像头的 center_y 偏移（更鲁棒版）。
改进点：
- 先做高斯模糊再阈值，减噪。
- 面积门槛调低到 80，且可调整。
- 基于当前中位数做 0.05 的离群剔除，防止偶发噪点。
- 打印中位数、均值、标准差，并给出末尾 10 帧的中位数，便于评估稳定度。
使用方法：保持标记物在原摄像头中心不动，运行 python temp/1.py，取 median 作为 y_offset（或均值若更平稳）。
"""
import time
import statistics
from typing import Tuple
from pathlib import Path
import sys

import cv2
import numpy as np

# 将项目根目录加入 sys.path，确保可以导入 src
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.vision import InferCamera

# === 配置：颜色阈值（默认亮橙），最小面积像素 ===
COLOR_LOWER: Tuple[int, int, int] = (10, 120, 70)
COLOR_UPPER: Tuple[int, int, int] = (25, 255, 255)
MIN_AREA: int = 80   # 调高可过滤噪声，调低可让小标记被检测
TARGET_HITS: int = 60  # 收集有效检测数量；可根据稳定度调整
OUTLIER_THR: float = 0.05  # 相对当前中位数的离群阈值（归一化坐标）


def main() -> None:
    cam = InferCamera()
    if not cam.open() or not cam.test_opened():
        print("[ERR] 摄像头打开失败")
        return
    print("[OK ] 摄像头已打开，开始采集。请保持标记物在原摄像头中心位置不动。")

    hits = []
    frame_id = 0
    try:
        while True:
            frame_id += 1
            ret, frame = cam.read()
            if not ret or frame is None:
                if frame_id % 10 == 0:
                    print("[WARN] 读取失败，重试中...")
                time.sleep(0.02)
                continue

            # 先模糊降噪，再转 HSV 阈值
            blurred = cv2.GaussianBlur(frame, (5, 5), 0)
            h, w = blurred.shape[:2]
            hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, COLOR_LOWER, COLOR_UPPER)
            mask = cv2.erode(mask, None, iterations=1)
            mask = cv2.dilate(mask, None, iterations=2)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                if frame_id % 15 == 0:
                    print("[INFO] 未检测到标记，请确认颜色/光照或调整 HSV 范围")
                continue

            contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(contour)
            if area < MIN_AREA:
                if frame_id % 15 == 0:
                    print(f"[INFO] 检测到面积过小({area:.1f}), 请让标记更大/更近或调低 MIN_AREA")
                continue

            m = cv2.moments(contour)
            if m["m00"] == 0:
                continue
            cx = m["m10"] / m["m00"]
            cy = m["m01"] / m["m00"]

            norm_y = cy / h

            # 离群剔除：相对当前中位数偏差过大则跳过
            if hits:
                med_now = statistics.median(hits)
                if abs(norm_y - med_now) > OUTLIER_THR:
                    if frame_id % 10 == 0:
                        print(f"[SKIP] 离群: center_y={norm_y:.4f}, median={med_now:.4f}")
                    continue

            hits.append(norm_y)
            running_avg = statistics.fmean(hits)
            med = statistics.median(hits)
            print(f"[HIT] frame={frame_id:04d} center_y={norm_y:.4f} running_avg={running_avg:.4f} median={med:.4f} samples={len(hits)}")

            if len(hits) >= TARGET_HITS:
                break

    except KeyboardInterrupt:
        print("[INFO] 手动中断，输出当前统计结果")
    finally:
        cam.close()

    if hits:
        avg = statistics.fmean(hits)
        med = statistics.median(hits)
        std = statistics.pstdev(hits) if len(hits) > 1 else 0.0
        tail = hits[-10:]
        tail_med = statistics.median(tail)
        tail_std = statistics.pstdev(tail) if len(tail) > 1 else 0.0
        print(f"[RESULT] 收集到 {len(hits)} 次有效检测")
        print(f"  平均值 running_avg = {avg:.4f}")
        print(f"  中位数 median      = {med:.4f}")
        print(f"  标准差 std         = {std:.4f}")
        print(f"  尾部10帧中位数     = {tail_med:.4f}, std = {tail_std:.4f}")
        print("请将中位数或尾部中位数作为 y_offset: delta_y = (center_y - 0.5) - y_offset")
    else:
        print("[FAIL] 没有得到有效检测，请调整 HSV 范围或标记物颜色/大小后重试")


if __name__ == "__main__":
    main()
