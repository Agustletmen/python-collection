import cv2
import dlib
import os
import numpy as np
from typing import Dict, List, Tuple
import time  # 顶部导入 time 模块（如果没有的话）

# -------------------------- 初始化核心模型（特征提取+检测） --------------------------
# 1. 人脸检测器（原有）
detector = dlib.get_frontal_face_detector()
# 2. 人脸关键点检测器（需下载模型文件）
predictor_path = "shape_predictor_68_face_landmarks.dat"
# 3. 人脸特征提取模型（需下载模型文件）
face_rec_model_path = "dlib_face_recognition_resnet_model_v1.dat"

# 检查模型文件是否存在（不存在则提示下载）
for model_path in [predictor_path, face_rec_model_path]:
    if not os.path.exists(model_path):
        print(f"错误：未找到模型文件 {model_path}")
        print("请从 dlib 官方仓库下载：https://github.com/davisking/dlib-models")
        exit()

predictor = dlib.shape_predictor(predictor_path)
face_recognizer = dlib.face_recognition_model_v1(face_rec_model_path)

# -------------------------- 配置参数 --------------------------
save_path = "captured_faces"  # 抓拍人脸保存目录
face_db_path = "face_database"  # 人脸库目录（存储已知人脸图像）
os.makedirs(save_path, exist_ok=True)
os.makedirs(face_db_path, exist_ok=True)

capture_mode = "video"  # 模式："image"（单图）或 "video"（视频/摄像头）
image_path = "test.jpg"  # 单图路径（仅 image 模式使用）
video_source = 0  # 视频源：0=本地摄像头，或视频文件路径（如 "test.mp4"）
similarity_threshold = 0.6  # 余弦相似度阈值（≤0.6 判定为同一人，可调整）

# -------------------------- 人脸库管理（核心数据结构） --------------------------
# 人脸库：key=人员ID，value=(特征向量, 姓名, 人脸图像路径)
face_database: Dict[str, Tuple[np.ndarray, str, str]] = {}
# 自动加载已存在的人脸库（若之前保存过）
if os.path.exists("face_db.npy"):
    face_database = np.load("face_db.npy", allow_pickle=True).item()
    print(f"成功加载人脸库，共 {len(face_database)} 人")


# -------------------------- 核心工具函数 --------------------------
def extract_face_feature(image: np.ndarray, face: dlib.rectangle) -> np.ndarray:
    """提取单张人脸的128维特征向量"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 获取68点关键点
    shape = predictor(gray, face)
    # 提取特征向量（128维）
    face_feature = face_recognizer.compute_face_descriptor(image, shape)
    return np.array(face_feature)


def compare_face_features(new_feature: np.ndarray) -> Tuple[bool, str, str]:
    """对比新特征与人脸库，返回 (是否重复, 人员ID, 人员姓名)"""
    if len(face_database) == 0:
        return False, "", ""  # 人脸库为空，无重复

    min_similarity = 1.0  # 余弦相似度越小，越相似（初始设为最大值1）
    matched_id = ""
    matched_name = ""

    for person_id, (db_feature, name, _) in face_database.items():
        # 计算余弦相似度（dlib 特征向量已归一化，直接点积即为相似度）
        similarity = np.dot(new_feature, db_feature)
        if similarity < min_similarity:
            min_similarity = similarity
            matched_id = person_id
            matched_name = name

    # 相似度低于阈值，判定为重复
    if min_similarity <= similarity_threshold:
        return True, matched_id, matched_name
    return False, "", ""


def add_face_to_db(face_feature: np.ndarray, face_roi: np.ndarray, name: str = "Unknown") -> str:
    """新增人脸到人脸库，返回人员ID"""
    # 用 time.time() 替代 os.time()，避免 os 模块问题
    person_id = f"PERSON_{len(face_database) + 1}_{int(time.time())}"
    db_img_path = os.path.join(face_db_path, f"{person_id}_{name}.jpg")
    cv2.imwrite(db_img_path, face_roi)
    face_database[person_id] = (face_feature, name, db_img_path)
    np.save("face_db.npy", face_database)
    print(f"新增人脸到库：ID={person_id}, 姓名={name}, 路径={db_img_path}")
    return person_id

# -------------------------- 抓拍核心函数（优化后） --------------------------



def capture_faces(image: np.ndarray, frame_id: int = 0) -> np.ndarray:
    """人脸检测+重复判断+抓拍/展示"""
    # 新增图像有效性判断
    if image is None or image.shape[0] == 0 or image.shape[1] == 0:
        print("警告：输入图像为空，跳过检测")
        return image

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if gray is None:
        print("警告：灰度图转换失败，跳过检测")
        return image

    faces = detector(gray, 1)  # 执行人脸检测

    for idx, face in enumerate(faces):
        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        face_roi = image[max(0, y1):min(y2, image.shape[0]), max(0, x1):min(x2, image.shape[1])]

        try:
            # 新增人脸ROI有效性判断（避免过小）
            if face_roi.shape[0] < 80 or face_roi.shape[1] < 80:
                print("警告：人脸区域过小（<80x80），跳过特征提取")
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(image, "Too Small", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                continue

            face_feature = extract_face_feature(image, face)
            is_duplicate, person_id, person_name = compare_face_features(face_feature)

            if is_duplicate:
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                info_text = f"Matched: {person_name} (ID:{person_id})"
                cv2.putText(image, info_text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                print(f"检测到重复人脸：{info_text}")
            else:
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                capture_name = f"frame_{frame_id}_face_{idx + 1}.jpg"
                capture_path = os.path.join(save_path, capture_name)
                cv2.imwrite(capture_path, face_roi)
                add_face_to_db(face_feature, face_roi)
                cv2.putText(image, "New Face Captured", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                print(f"新增抓拍人脸：{capture_path}")

        except Exception as e:
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(image, "Feature Extract Failed", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            print(f"人脸特征提取失败：{str(e)}")

    return image


# -------------------------- 执行抓拍（原有逻辑不变） --------------------------
if capture_mode == "image":
    img = cv2.imread(image_path)
    if img is None:
        print("错误：未找到图像，请检查路径！")
        exit()
    result_img = capture_faces(img)
    cv2.imshow("Face Capture Result", result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

elif capture_mode == "video":
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("错误：无法打开视频源！")
        exit()

    # 可选：设置视频分辨率（降低分辨率提升速度）
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result_frame = capture_faces(frame, frame_id)
        cv2.imshow("Real-Time Face Capture (No Duplicate)", result_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        frame_id += 1

    cap.release()
    cv2.destroyAllWindows()

print("抓拍完成！")
print(f"人脸库统计：共 {len(face_database)} 人，存储路径：{face_db_path}")