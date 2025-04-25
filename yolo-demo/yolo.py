import cv2
from ultralytics import YOLO
import torch

# 检查是否有可用的 GPU（YOLO自身也会优先使用cuda，无需额外调用，这里打印出来方便查看）
print("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load the YOLO model
model_path = "./yolov8n.pt" # 对象检测
# model_path = "./yolov8n-cls.pt"# 图像分类
# model_path = "./yolov8n-pose.pt"# 姿态估计
# model_path = "./yolov8n-seg.pt"# 图像分割
model = YOLO(model=model_path).to(device) 

# Initialize the video capture from the default camera (usually camera index 0)
cap = cv2.VideoCapture(0)

while True:
    # ret=true|flase   frame=numpy.array
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame")
        break
    
    # 对每一帧执行推理（inference），获取检测结果
    results = model(frame)[0]
    
    # 在原始帧上绘制检测结果。这包括检测框、类别标签和置信度分数。
    annotated_frame = results.plot()
    
    # 显示带有检测结果的图像
    cv2.imshow('YOLOv8 Inference', annotated_frame)
    
    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()