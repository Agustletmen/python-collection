from ultralytics import YOLO

# 加载预训练模型
model = YOLO("yolov8n.pt")

# 训练模型
results = model.train(
    data="data.yaml",
    epochs=100,
    device=0,
    name="custom_train"
)