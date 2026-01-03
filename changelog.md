# 更新日志

## 2026-01-03
- 调整 PyCoral 跑分脚本的后处理逻辑：假定模型输出固定为 [1, 6, N]，字段顺序为 [x, y, w, h, score, cls_id]，去掉其他兼容分支，确保当前双标签模型的 cls_id 解析正确。
- 影响：如果后续更换为不同输出格式的模型，需要同步更新解析逻辑；目前模型正常则无需额外操作。
- 新增测试工具 tests/run_single_infer.py：启动摄像头读取一帧，使用 PyCoral (Edge TPU) 进行单次推理，输出模型输入输出信息与解析后的检测结果，便于快速验证模型行为。
- 新增实时推理工具 tests/run_stream_infer.py：循环读取摄像头帧，使用 PyCoral Edge TPU 推理，并通过 ANSI 重绘在终端实时显示 FPS、单帧耗时和检测结果（Ctrl+C 退出）。
- 优化 tests/run_stream_infer.py：当摄像头分辨率与模型输入一致时跳过 resize，只做 BGR→RGB；终端每隔固定帧数 (默认5) 才重绘，降低 I/O 开销以提升 FPS。
- 再次优化 tests/run_stream_infer.py：使用滑动窗口计算短期 FPS，减少累计平均带来的漂移，使显示的 FPS 更接近实时表现。
- 进一步优化 tests/run_stream_infer.py：
	- 按时间间隔（默认 0.3s）重绘而非每 N 帧，减少终端 I/O。
	- 仅在间隔帧做后处理解析，其余帧只做推理，降低 CPU 开销。
	- 预取输出张量索引与量化参数，避免循环内重复查询。
