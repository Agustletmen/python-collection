import cv2
import mediapipe as mp
"""
面部关键点检测可识别面部的 468 个关键点，常用于表情识别、美颜等场景。
"""
# 初始化 MediaPipe 面部检测模块
mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh()
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)


# 打开摄像头
cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture("rtsp://admin:xirui360@192.168.3.136:554/Streaming/Channels/101")
while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("无法读取摄像头画面")
        continue

    # 转换颜色空间
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = face_mesh.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # 绘制面部关键点和连接线
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACE_CONNECTIONS,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec)
    cv2.imshow('MediaPipe Face Mesh', image)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()    