import cv2
import mediapipe as mp

# 初始化objectron模块
mp_objectron = mp.solutions.objectron
objectron = mp_objectron.Objectron(
    static_image_mode=False,
    max_num_objects=5,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.99,
    model_name='Shoe'
)

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

    # 进行3D物体检测
    results = objectron.process(image)

    # 将图像转换回BGR格式
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # 如果检测到物体
    if results.detected_objects:
        for detected_object in results.detected_objects:
            # 绘制物体的3D边界框
            mp_drawing.draw_landmarks(
                image,
                detected_object.landmarks_2d,
                mp_objectron.BOX_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
            )

    # 显示图像
    cv2.imshow('Objectron Detection', image)

    # 按 'q' 键退出循环
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# 释放摄像头并关闭窗口
cap.release()
cv2.destroyAllWindows()
