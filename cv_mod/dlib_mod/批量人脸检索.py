import os
import insightface
import cv2
import numpy as np

app = FaceAnalysis(providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# 1. 构建图库特征库（读取 gallery 目录下所有图片）
gallery_path = "gallery"  # 图库目录
gallery_features = []  # 存储特征向量
gallery_names = []     # 存储图片名称

for img_name in os.listdir(gallery_path):
    img_path = os.path.join(gallery_path, img_name)
    img = cv2.imread(img_path)
    faces = app.get(img)
    if len(faces) > 0:
        gallery_features.append(faces[0].embedding)
        gallery_names.append(img_name)

# 2. 提取目标人脸特征
target_img = cv2.imread("target.jpg")  # 目标人脸图片
target_faces = app.get(target_img)
if len(target_faces) == 0:
    print("目标图片未检测到人脸！")
else:
    target_feat = target_faces[0].embedding

    # 3. 计算目标与图库的相似度
    similarities = [np.dot(target_feat, feat)/(np.linalg.norm(target_feat)*np.linalg.norm(feat)) 
                    for feat in gallery_features]

    # 4. 排序并取 Top-3 最相似结果
    top_indices = np.argsort(similarities)[::-1][:3]
    print("Top-3 最相似人脸：")
    for idx in top_indices:
        print(f"图片名称：{gallery_names[idx]}, 相似度：{similarities[idx]:.4f}")