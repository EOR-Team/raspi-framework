"""EdgeTPU delegate self-test using pycoral.

- Verifies delegate loading via pycoral.utils.edgetpu.make_interpreter
- Runs warmup + timed inferences on dummy input matching model shape
- Reports latency and FPS
"""
import argparse
import sys
import time
from pathlib import Path
from typing import Tuple

import numpy as np
from pycoral.adapters import common
from pycoral.utils.edgetpu import make_interpreter

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config


def build_interpreter(model_path: str):
    try:
        interpreter = make_interpreter(model_path)
        interpreter.allocate_tensors()
        return interpreter
    except Exception as exc:  # pragma: no cover - runtime safety
        print(f"[ERROR] Failed to create interpreter with EdgeTPU delegate: {exc}")
        raise SystemExit(1)


def make_dummy_input(interpreter) -> Tuple[np.ndarray, dict]:
    input_details = interpreter.get_input_details()[0]
    shape = input_details["shape"]
    dtype = input_details["dtype"]
    data = np.zeros(shape, dtype=dtype)
    return data, input_details


def run_benchmark(interpreter, iterations: int, warmup: int) -> Tuple[float, float]:
    data, _ = make_dummy_input(interpreter)

    for _ in range(warmup):
        common.set_input(interpreter, data)
        interpreter.invoke()

    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        common.set_input(interpreter, data)
        interpreter.invoke()
        end = time.perf_counter()
        times.append(end - start)

    avg = sum(times) / len(times) if times else 0.0
    best = min(times) if times else 0.0
    return avg, best


def main() -> int:
    parser = argparse.ArgumentParser(description="EdgeTPU delegate self-test (pycoral)")
    parser.add_argument("--model", default=config.AIMBOT_MODEL_PATH, help="Path to EdgeTPU tflite model")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=50, help="Timed iterations")
    args = parser.parse_args()

    print(f"[INFO] Model: {args.model}")
    print("[INFO] Loading interpreter with EdgeTPU delegate via pycoral...")
    interpreter = build_interpreter(args.model)

    input_details = interpreter.get_input_details()[0]
    output_meta = interpreter.get_output_details()
    print(f"[INFO] Input shape: {input_details['shape']}, dtype: {input_details['dtype']}")
    for idx, od in enumerate(output_meta):
        print(f"[INFO] Output {idx}: shape={od['shape']} dtype={od['dtype']}")

    print(f"[INFO] Warmup: {args.warmup} iterations, Benchmark: {args.iters} iterations")
    avg, best = run_benchmark(interpreter, args.iters, args.warmup)
    fps_avg = 1.0 / avg if avg > 0 else 0.0
    fps_best = 1.0 / best if best > 0 else 0.0

    print(f"[RESULT] Avg latency: {avg*1000:.2f} ms ({fps_avg:.2f} FPS)")
    print(f"[RESULT] Best latency: {best*1000:.2f} ms ({fps_best:.2f} FPS)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
