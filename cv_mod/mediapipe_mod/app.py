import cv2
import mediapipe as mp
import time
import os
from datetime import datetime

# 初始化 MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(
    model_selection=1,  # 0 适用于近距离人脸，1 适用于最远 5 米的人脸
    min_detection_confidence=0.5  # 最小检测置信度
)

# 抓拍配置参数
CAPTURE_MODE = "crop"  # "full" 保存完整画面，"crop" 只保存人脸区域
COOLDOWN_TIME = 3  # 抓拍冷却时间（秒），避免连续抓拍
SAVE_DIR = "face_captures"  # 抓拍图片保存目录
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
FONT_COLOR = (0, 255, 0)  # 绿色文字
LINE_TYPE = 2

# 创建保存目录（如果不存在）
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# 打开摄像头
cap = cv2.VideoCapture(0)
last_capture_time = 0  # 上次抓拍时间
capture_count = 0  # 抓拍计数器

print(f"人脸抓拍程序已启动！")
print(f"保存模式：{CAPTURE_MODE}")
print(f"冷却时间：{COOLDOWN_TIME}秒")
print(f"图片保存目录：{os.path.abspath(SAVE_DIR)}")
print("按 'q' 退出程序，按 's' 手动抓拍")

def capture_face(image, face_landmarks, capture_mode="full"):
    """
    抓拍人脸并保存图片
    :param image: 原始图像
    :param face_landmarks: 人脸检测结果
    :param capture_mode: 保存模式 "full" 或 "crop"
    :return: 保存的文件名
    """
    global capture_count
    
    # 生成时间戳文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # 精确到毫秒
    capture_count += 1
    filename = f"face_{capture_count}_{timestamp}.jpg"
    save_path = os.path.join(SAVE_DIR, filename)
    
    if capture_mode == "crop":
        # 获取图像尺寸
        h, w, _ = image.shape
        
        # 获取人脸检测框的相对坐标
        bbox = face_landmarks.location_data.relative_bounding_box
        
        # 转换为绝对坐标（并添加边界缓冲）
        x1 = int(bbox.xmin * w - 20)
        y1 = int(bbox.ymin * h - 30)
        x2 = int((bbox.xmin + bbox.width) * w + 20)
        y2 = int((bbox.ymin + bbox.height) * h + 30)
        
        # 确保坐标不超出图像范围
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        
        # 裁剪人脸区域
        face_crop = image[y1:y2, x1:x2]
        
        # 保存裁剪后的人脸
        cv2.imwrite(save_path, face_crop)
    else:
        # 保存完整画面
        cv2.imwrite(save_path, image)
    
    print(f"已抓拍并保存：{filename}")
    return filename

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("无法获取摄像头画面")
        continue

    # 水平翻转图像（镜像效果，更自然）
    image = cv2.flip(image, 1)
    
    # 保存原始图像用于显示提示（避免在处理后的图像上重复绘制）
    image_display = image.copy()
    
    # 将 BGR 图像转换为 RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 进行人脸检测
    results = face_detection.process(image_rgb)
    
    # 检测到人脸
    if results.detections:
        for detection in results.detections:
            # 绘制人脸检测框和关键点
            mp_drawing.draw_detection(image_display, detection)
            
            # 自动抓拍（检查冷却时间）
            current_time = time.time()
            if current_time - last_capture_time > COOLDOWN_TIME:
                capture_face(image, detection, CAPTURE_MODE)
                last_capture_time = current_time
                
                # 在画面上显示抓拍提示
                cv2.putText(image_display, "Captured!", (50, 50), 
                           FONT, FONT_SCALE, (0, 0, 255), LINE_TYPE)
    
    # 显示统计信息
    info_text = f"Captures: {capture_count} | Mode: {CAPTURE_MODE} | Press 's' to capture"
    cv2.putText(image_display, info_text, (10, 30), 
               FONT, FONT_SCALE * 0.8, FONT_COLOR, 1)
    
    # 显示结果
    cv2.imshow('MediaPipe Face Detection & Capture', image_display)
    
    # 键盘控制
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        # 按 'q' 退出
        break
    elif key == ord('s'):
        # 按 's' 手动抓拍
        if results.detections:
            # 如果检测到人脸，抓拍第一个人脸
            capture_face(image, results.detections[0], CAPTURE_MODE)
            last_capture_time = time.time()
            
            # 显示手动抓拍提示
            temp_image = image_display.copy()
            cv2.putText(temp_image, "Manual Capture!", (50, 50), 
                       FONT, FONT_SCALE, (255, 0, 0), LINE_TYPE)
            cv2.imshow('MediaPipe Face Detection & Capture', temp_image)
            cv2.waitKey(500)  # 显示提示 500ms

# 释放资源
cap.release()
cv2.destroyAllWindows()
face_detection.close()  # 关闭 MediaPipe 检测器
print(f"\n程序已退出，共抓拍 {capture_count} 张人脸图片")
print(f"图片保存位置：{os.path.abspath(SAVE_DIR)}")