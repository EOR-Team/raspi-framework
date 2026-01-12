import cv2
import time
from robomaster import robot
from robomaster.gimbal import Gimbal
from robomaster.blaster import Blaster

from src.vision import InferCamera
from src import logger, config, uart
from src.vision.detect import YoloV8Detector

logger.set_debug_mode(False)

# ==== 摄像头 ====
logger.info("打开摄像头...")
cam = InferCamera()
if not cam.open() or not cam.test_opened():
    logger.error("摄像头打开失败")
    exit(1)
logger.info("摄像头打开成功")
logger.info(f"摄像头参数: {cam.get_actual_settings()}")

# ==== 串口握手（技能测试心跳） ====
uart.start_backend()
logger.info("连接串口...")
line = uart.readline()
if "aimbot start" not in line:
    logger.error("串口连接失败，请检查连接")
    exit(1)
logger.info("串口连接成功")

# ==== dji robot ====
ep_robot = robot.Robot()
logger.info("连接机器人...")
try:
    ep_robot.initialize(conn_type="rndis")
    version_info = ep_robot.get_version()
    time.sleep(5)  # 等待连接稳定
except:
    logger.error("机器人连接失败")
    uart.stop_backend()
    exit(1)
logger.info(f"机器人连接成功, 版本信息: {version_info}")

blaster = Blaster(ep_robot)
blaster.set_led(0)
gimbal = Gimbal(ep_robot)
gimbal.recenter().wait_for_completed()
gimbal.moveto(-25, 0, 180, 180).wait_for_completed()

# ==== 检测模型 ====
logger.info("加载检测模型...")
detector = YoloV8Detector(
    config.AIMBOT_MODEL_PATH
)
if not detector.initialize():
    logger.error("检测模型加载失败")
    exit(1)
logger.info("检测模型加载成功")


not_detected_time = 0
last_fire_time = 0.0
fire_interval = 0.3
frame_counter = 0
in_deadzone_counter = 0
last_alive_time = time.time()

# ==== 主循环 ====
while True:
    try:
        start_time = time.perf_counter()

        ret, frame = cam.read()
        if not ret or frame is None:
            logger.warning("摄像头读取失败")
        frame_counter += 1
        if frame_counter % 5 == 0:
            cv2.imwrite("temp/current.jpg", frame)
        
        detections, infer_ms = detector.invoke(frame)

        # 当前默认只处理第一个目标
        if len(detections) == 0:
            logger.info("未检测到目标")
            not_detected_time += 1
            if not_detected_time >= 30: # TODO: 测试用 30
                logger.info("长时间未检测到目标，云台复位 PID重置")
                gimbal.recenter().wait_for_completed()
                gimbal.moveto(-25, 0, 180, 180).wait_for_completed()
                config.AIMBOT_HOR_PID.reset()
                config.AIMBOT_VER_PID.reset()
                not_detected_time = 0
            continue
        else:
            not_detected_time = 0
            logger.info(f"检测到 {len(detections)} 个目标")

        detect = detections[0]
        logger.info(f"检测到目标: {detect}, 推理时间: {infer_ms}ms")

        # PID瞄准
        center_x = detect.xyxyn[0] + detect.xyxyn[2]
        center_x /= 2.0
        center_y = detect.xyxyn[1] + detect.xyxyn[3]
        center_y /= 2.0

        delta_x = center_x - config.CROSSHAIR_X
        # 垂直方向补偿相机抬高偏移
        delta_y = center_y - config.CROSSHAIR_Y
        logger.info(f"delta_x: {delta_x}, delta_y: {delta_y}")

        # 检查是否进入死区，在死区内尝试开火，2s 最多一次
        # in_deadzone = (
        #     config.AIMBOT_DEADZONE_X_MIN <= delta_x <= config.AIMBOT_DEADZONE_X_MAX and
        #     config.AIMBOT_DEADZONE_Y_MIN <= delta_y <= config.AIMBOT_DEADZONE_Y_MAX
        # )
        in_deadzone = (
            abs(delta_x) < config.AIMBOT_DEADZONE_X_SIZE and
            abs(delta_y) < config.AIMBOT_DEADZONE_Y_SIZE
        )

        now = time.perf_counter()
        if in_deadzone:
            in_deadzone_counter += 1
        else:
            in_deadzone_counter = 0

        if in_deadzone_counter >= 5 and now - last_fire_time >= fire_interval:
            try:
                gimbal.drive_speed(0, 0)  # 停云台
                blaster.fire()
                last_fire_time = now
                logger.info("目标在死区内，开火一次")
            except Exception as e:
                logger.error(f"开火失败: {e}")

        # PID 输出视为角速度指令，限制在配置的 U_MAX 内
        ctrl_x = config.AIMBOT_HOR_PID(delta_x)
        ctrl_y = config.AIMBOT_VER_PID(delta_y)

        u_max = getattr(config, "U_MAX", 180.0)
        ctrl_x = max(-u_max, min(u_max, ctrl_x))
        ctrl_y = max(-u_max, min(u_max, ctrl_y))

        # 直接下发速度指令，符号与画面坐标系相反（俯仰轴方向与水平相反）
        gimbal.drive_speed(
            pitch_speed=ctrl_y,
            yaw_speed=-ctrl_x,
        )
        logger.info(f"云台控制命令: pitch={ctrl_y}, yaw={-ctrl_x}")

        # 保持循环节奏：测量本轮耗时，按目标周期补足
        end_time = time.perf_counter()
        elapsed = end_time - start_time
        remain = config.AIMBOT_ACTION_DELAY - elapsed
        delay = remain if remain > 0 else 0.0
        line = uart.readline(timeout=delay)
        if len(line) > 0:
            last_alive_time = time.time()
        if time.time() - last_alive_time > 2.0:
            logger.error("与技能测试器失去连接")
            raise KeyboardInterrupt()
        
    except KeyboardInterrupt:
        logger.info("程序终止")
        # gimbal.reset()
        # gimbal.recenter().wait_for_completed()
        import cv2
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        gimbal.drive_speed(0, 0)
        input_str = input("Press Y to exit...")
        if input_str.lower() == 'y':
            cam.close()
            gimbal.drive_speed(0, 0)
            gimbal.recenter().wait_for_completed()
            ep_robot.close()
            uart.stop_backend()
            exit(0)
        else:
            logger.info("继续运行程序")
            from src import config # 重新加载配置 应该不行。
            continue
