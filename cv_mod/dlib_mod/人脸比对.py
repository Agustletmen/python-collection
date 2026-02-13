import insightface
from insightface.app import FaceAnalysis
import numpy as np

# 1. 初始化模型（同上）
app = FaceAnalysis(providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# 2. 读取两张待比对图片（face1.jpg 和 face2.jpg）
img1 = cv2.imread("face1.jpg")
img2 = cv2.imread("face2.jpg")

# 3. 提取两张图片的人脸特征（假设每张图只有 1 个人脸）
faces1 = app.get(img1)
faces2 = app.get(img2)
if len(faces1) == 0 or len(faces2) == 0:
    print("其中一张图片未检测到人脸！")
else:
    feat1 = faces1[0].embedding  # 第一张图的人脸特征
    feat2 = faces2[0].embedding  # 第二张图的人脸特征

    # 4. 计算余弦相似度（越接近 1 越相似）
    similarity = np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2))
    print(f"两张人脸的相似度：{similarity:.4f}")

    # 5. 判定是否为同一人（阈值设为 0.65）
    threshold = 0.65
    if similarity >= threshold:
        print("判定为：同一人")
    else:
        print("判定为：不同人")