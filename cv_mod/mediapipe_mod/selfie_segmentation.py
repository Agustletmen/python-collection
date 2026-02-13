import cv2
import mediapipe as mp
import numpy as np

# 初始化selfie_segmentation模块
mp_selfie_segmentation = mp.solutions.selfie_segmentation
selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

# 打开摄像头
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("无法读取摄像头数据。")
        continue

    # 将图像转换为RGB格式
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 进行人像分割
    results = selfie_segmentation.process(image)

    # 将图像转换回BGR格式
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # 获取分割掩码
    mask = results.segmentation_mask

    # 创建一个纯色背景图像
    background = np.zeros(image.shape, dtype=np.uint8)
    background[:] = (0, 255, 0)  # 绿色背景

    # 根据分割掩码将前景和背景合并
    condition = np.stack((mask,) * 3, axis=-1) > 0.1
    output_image = np.where(condition, image, background)

    # 显示图像
    cv2.imshow('Selfie Segmentation', output_image)

    # 按 'q' 键退出循环
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# 释放摄像头并关闭窗口
cap.release()
cv2.destroyAllWindows()
