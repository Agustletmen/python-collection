import torch
from ultralytics import YOLO

# 检查是否有可用的 GPU（YOLO自身也会优先使用cuda，无需额外调用，这里打印出来方便查看）
print("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the YOLO model
# 主要是v8、v11比较常用，版本高不一定代表模型更好
# model_path = "./yolov8n.pt"  # 对象检测
# model_path = "./yolov8n-obb.pt" # 旋转对象检测
# model_path = "./yolov8n-cls.pt"# 图像分类
model_path = "./yolov8n-pose.pt"# 姿态估计
# model_path = "./yolov8n-seg.pt"# 图像分割
model = YOLO(model=model_path).to(device)

print(model.task)  # 不同的模型对应不同的任务
print(model.names)  # 同一任务能识别的类型也不同
print(sum(p.numel() for p in model.parameters()))  # 模型参数数量，# n nano、s small、m medium、l large、x extra-large

model.predict(
    source=0,
    show=True,
    save=False,
)

# # Initialize the video capture from the default camera (usually camera index 0)
# cap = cv2.VideoCapture(0)
#
# while True:
#     # ret=true|flase   frame=numpy.array
#     ret, frame = cap.read()
#
#     if not ret:
#         print("Failed to grab frame")
#         break
#
#     # 对每一帧执行推理（inference），获取检测结果
#     results = model(frame)[0]
#
#     # 在原始帧上绘制检测结果。这包括检测框、类别标签和置信度分数。
#     annotated_frame = results.plot()
#
#     # 显示带有检测结果的图像
#     cv2.imshow('YOLOv8 Inference', annotated_frame)
#
#     # Break the loop if 'q' key is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()
