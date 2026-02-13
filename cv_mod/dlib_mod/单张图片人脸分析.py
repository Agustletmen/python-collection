import insightface
from insightface.app import FaceAnalysis
import cv2
import time

# 1. 初始化 FaceAnalysis（GPU 加速，适配 GTX 1050）
# 若 GPU 运行报错，可切换为 providers=['CPUExecutionProvider']
# app = FaceAnalysis(providers=['CPUExecutionProvider'])
app = FaceAnalysis(
    name='buffalo_s',  # 轻量模型（默认是buffalo_l，不用改）
    providers=['CUDAExecutionProvider'],
    allowed_modules=['detection', 'recognition']
)
# 调整 det_size 平衡速度与精度（GTX 1050 建议 640x640，显存紧张可改为 480x480）
app.prepare(ctx_id=0, det_size=(320, 320))
print("模型初始化完成，开启实时流检测...")
print("按 'q' 或 ESC 键退出程序")
# print("GPU是否可用:", insightface.utils.is_available('cuda'))
print("当前使用的Provider:", app.runtime_providers)

# 2. 打开摄像头（0 为默认摄像头，外接摄像头可改为 1/2 等）
cap = cv2.VideoCapture(0)
# 设置摄像头分辨率（根据硬件支持调整，提升性能可降低分辨率）
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# 检查摄像头是否正常打开
if not cap.isOpened():
    print("错误：无法打开摄像头！")
    exit()

# 3. 实时流处理循环
fps_list = []  # 用于计算平均 FPS
while True:
    # 读取一帧画面
    ret, frame = cap.read()
    if not ret:
        print("警告：无法读取摄像头画面，可能已断开连接！")
        break

    # 记录帧处理开始时间（用于计算 FPS）
    start_time = time.time()

    # 4. 人脸分析（核心：检测+属性提取）
    faces = app.get(frame)  # 对当前帧执行人脸分析

    # 5. 实时可视化（在帧上画人脸框+属性文字）
    for face in faces:
        # 画人脸框（绿色，线宽 2）
        bbox = face.bbox.astype(int)  # 坐标转为整数
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

        embedding = face.normed_embedding # 人脸特征数据

        # 叠加年龄+性别文字（蓝色）
        # age_gender = f"{face.age:.0f}岁 {'男' if face.sex == 'Male' else '女'}"
        # cv2.putText(frame, age_gender, (bbox[0], bbox[1]-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # 叠加表情文字（红色）
        # emotion = face.emotion
        # cv2.putText(frame, emotion, (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # 计算 FPS（帧率）
    fps = 1 / (time.time() - start_time)
    fps_list.append(fps)
    # 显示平均 FPS（取最近 10 帧的平均值，更稳定）
    avg_fps = sum(fps_list[-10:]) / len(fps_list[-10:])
    # cv2.putText(frame, f"FPS: {avg_fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)

    # 显示提示文字
    # cv2.putText(frame, "Press 'q' or ESC to exit", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 1)

    # 6. 显示处理后的画面
    cv2.imshow("Real-Time Face Analysis (InsightFace)", frame)

    # 7. 退出控制（按 'q' 或 ESC 键退出）
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:  # 27 是 ESC 键的 ASCII 码
        print("正在退出程序...")
        break

# 8. 释放资源（必须执行，避免摄像头占用）
cap.release()
cv2.destroyAllWindows()
print("程序已退出！")