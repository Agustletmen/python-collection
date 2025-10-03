import cv2
import mediapipe as mp

# 初始化holistic模块
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic()

# 初始化绘图工具
mp_drawing = mp.solutions.drawing_utils

# 打开摄像头
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("无法读取摄像头数据。")
        continue

    # 将图像转换为RGB格式
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 进行检测
    results = holistic.process(image)

    # 将图像转换回BGR格式
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # 绘制人脸关键点
    if results.face_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.face_landmarks,
            mp_holistic.FACEMESH_TESSELATION,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1)
        )

    # 绘制姿势关键点
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
        )

    # 绘制左手关键点
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
        )

    # 绘制右手关键点
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
        )

    # 显示图像
    cv2.imshow('Holistic Detection', image)

    # 按 'q' 键退出循环
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# 释放摄像头并关闭窗口
cap.release()
cv2.destroyAllWindows()
