"""Benchmark Coral USB Accelerator throughput with AimbotDetector and InferCamera."""
import argparse
import gc
import sys
import time
from pathlib import Path
from typing import Optional

# Ensure project root on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.logger import get_logger  # noqa: E402
from src.vision.camera import InferCamera  # noqa: E402
from src.vision.detector import AimbotDetector  # noqa: E402


logger = get_logger()


def warmup(detector: AimbotDetector, camera: InferCamera, warmup_frames: int) -> None:
    completed = 0
    while completed < warmup_frames:
        ret, frame = camera.read()
        if not ret or frame is None:
            continue
        detector.detect(frame)
        completed += 1
    logger.info(f"Warmup completed: {completed} frames")


def benchmark(detector: AimbotDetector, camera: InferCamera, duration: float, max_frames: Optional[int]) -> dict:
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
    parser = argparse.ArgumentParser(description="Benchmark Coral USB Accelerator with AimbotDetector")
    parser.add_argument("--duration", type=float, default=10.0, help="Benchmark duration in seconds")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup frames before timing")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional cap on processed frames")
    args = parser.parse_args()

    camera = InferCamera()
    detector: Optional[AimbotDetector] = None

    try:
        if not camera.open():
            logger.error("Failed to open InferCamera")
            return 1

        actual = camera.get_actual_settings()
        if actual:
            logger.info(
                f"Camera opened: {actual['width']}x{actual['height']}@{actual['fps']}FPS fourcc={actual['fourcc']}"
            )

        detector = AimbotDetector()
        if not detector.initialize():
            logger.error("Failed to initialize AimbotDetector")
            return 1

        logger.info("Starting warmup...")
        warmup(detector, camera, args.warmup)

        logger.info("Starting benchmark...")
        stats = benchmark(detector, camera, args.duration, args.max_frames)

        logger.info(
            f"Frames: {stats['frames']} | Elapsed: {stats['elapsed']:.3f}s | FPS: {stats['fps']:.2f} | "
            f"Total detections: {stats['detections']}"
        )

        return 0

    finally:
        # Best-effort cleanup to avoid native teardown crashes
        try:
            camera.close()
        except Exception:
            pass

        if detector is not None:
            try:
                detector.model = None
            except Exception:
                pass

        gc.collect()


if __name__ == "__main__":
    raise SystemExit(main())
