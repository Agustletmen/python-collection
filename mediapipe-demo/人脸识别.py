import cv2
import mediapipe as mp

# 初始化 MediaPipe 人脸检测模块
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# 设定人脸检测参数
# model_selection: 0 适用于距离较近的人脸（2 米内），1 适用于距离较远的人脸（5 米内）
# min_detection_confidence: 最小检测置信度，高于该值才认为检测到人脸
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

# 打开摄像头
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("无法读取摄像头画面")
        continue

    # 转换颜色空间，因为 MediaPipe 要求输入为 RGB 格式
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    # 进行人脸检测
    results = face_detection.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.detections:
        for detection in results.detections:
            # 绘制人脸检测框和关键点
            mp_drawing.draw_detection(image, detection)
    # 显示处理后的图像
    cv2.imshow('MediaPipe Face Detection', image)
    # 按 'Esc' 键退出循环
    if cv2.waitKey(5) & 0xFF == 27:
        break

# 释放摄像头资源
cap.release()
# 关闭所有 OpenCV 窗口
cv2.destroyAllWindows()
    