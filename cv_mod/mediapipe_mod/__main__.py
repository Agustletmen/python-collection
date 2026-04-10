from pathlib import Path

import cv2
import mediapipe
from mediapipe.tasks.python import vision, text, audio
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.core import vision_task_running_mode

"""
使用的是google自研的BlazeFace、BlazePose等模型

drawing_styles
drawing_utils


😀人脸相关
FaceDetector
FaceDetectorOptions
FaceDetectorResult

FaceLandmarker
FaceLandmarkerOptions
FaceLandmarkerResult
FaceLandmarksConnections

✋手势识别
GestureRecognizer
GestureRecognizerOptions
GestureRecognizerResult

🖐️手部相关
HandLandmarker
HandLandmarkerOptions
HandLandmarkerResult
HandLandmarksConnections

HolisticLandmarker
HolisticLandmarkerOptions
HolisticLandmarkerResult

ImageClassifier
ImageClassifierOptions
ImageClassifierResult

ImageEmbedder
ImageEmbedderOptions
ImageEmbedderResult

ImageSegmenter
ImageSegmenterOptions
ImageProcessingOptions

交互式分割
InteractiveSegmenter
InteractiveSegmenterOptions
InteractiveSegmenterRegionOfInterest

目标检测
ObjectDetector
ObjectDetectorOptions
ObjectDetectorResult

🧍人体姿态
PoseLandmark
PoseLandmarker
PoseLandmarkerOptions
PoseLandmarkerResult
PoseLandmarksConnections

RunningMode
"""

model_dir = "./models"

# 绘图
mp_drawing = vision.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1) # 画笔样式配置

# 对象检测
objectDetector = vision.ObjectDetector.create_from_options(
    vision.ObjectDetectorOptions(
        base_options=BaseOptions(model_asset_path=None),
        max_results=5,
        score_threshold=0.5
    )
)

# 图像分类
imageClassifier = vision.ImageClassifier.create_from_options(
    vision.ImageClassifierOptions(
        base_options=BaseOptions(model_asset_path=None),
        max_results=5,
        score_threshold=0.5
    )
)

# 图片分割，提供轻量级人像分割模型，快速分离图像中的前景（人体）和背景。
imageSegmenter = vision.ImageSegmenter.create_from_options(
    vision.ImageSegmenterOptions(
        base_options=BaseOptions(model_asset_path=None),
        output_category_mask=True,
        output_confidence_masks=False
    )
)

# 交互式分割
interactiveSegmenter = vision.InteractiveSegmenter.create_from_options(
    vision.InteractiveSegmenterOptions(
        base_options=BaseOptions(model_asset_path=None),
        output_confidence_masks=True,
        output_category_mask=True
    )
)

# 图片嵌入
imageEmbedder = vision.ImageEmbedder.create_from_options(
    vision.ImageEmbedderOptions(
        base_options=BaseOptions(model_asset_path=None)
    )
)

# 人脸检测、6 个面部关键点（如双眼中心、鼻尖）及置信度。
faceDetector = vision.FaceDetector.create_from_options(
    vision.FaceDetectorOptions(
        base_options=BaseOptions(model_asset_path=None),
        min_suppression_threshold=0.5  # 最小检测置信度
    )
)

# 人脸特征点检测，可检测 468 个精细的 3D 面部关键点
faceLandmarker = vision.FaceLandmarker.create_from_options(
    vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=None),
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
)

# 手部特征点检测，提供手部检测与关键点识别模型，可检测单 / 双手的 21 个 3D 关键点
handLandmarker = vision.HandLandmarker.create_from_options(
    vision.HandLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path="D:\\project\\python-collection\\cv_mod\\mediapipe_mod\\models\\hand_landmarker.task"
        ),
        running_mode=vision_task_running_mode.VisionTaskRunningMode.LIVE_STREAM,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
)

# 手势识别
gestureRecognizer = vision.GestureRecognizer.create_from_options(
    vision.GestureRecognizerOptions(
        base_options=BaseOptions(model_asset_path=None),
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
)

# 姿势特征点检测，提供人体姿态检测模型，专注于身体关键关节点（如头部、肩膀、手肘、膝盖等 33 个 3D 关键点）的检测与跟踪。
poseLandmarker = vision.PoseLandmarker.create_from_options(
    vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=None),
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
)

# 整体特征点检测，整合了面部网格、手部关键点、身体姿态三种检测能力。
holisticLandmarker = vision.HolisticLandmarker.create_from_options(
    vision.HolisticLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=None),
        min_face_detection_confidence=0.5,
        min_pose_detection_confidence=0.5,
    )
)

# 语言检测
language_detector = text.LanguageDetector.create_from_options(
    text.LanguageDetectorOptions(
        base_options=BaseOptions(model_asset_path=None)
    )
)

# 文本分类
text_classifier = text.TextClassifier.create_from_options(
    text.TextClassifierOptions(
        base_options=BaseOptions(model_asset_path=None)
    )
)

# 文本嵌入
text_embedder = text.TextEmbedder.create_from_options(
    text.TextEmbedderOptions(
        base_options=BaseOptions(model_asset_path=None)
    )
)

# 音频分类
audio_classifier = audio.AudioClassifier.create_from_options(
    audio.AudioClassifierOptions(
        base_options=BaseOptions(model_asset_path=None)
    )
)


def main():
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("无法获取摄像头画面")
            continue

        # 水平翻转图像（镜像效果，更自然）
        image = cv2.flip(image, 1)

        # 将 BGR 图像转换为 RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 将图像转换为 MediaPipe 图像对象
        mp_image = mediapipe.Image(image_format=mediapipe.ImageFormat.SRGB, data=image_rgb)

        handLandmarker.detect_async(mp_image, timestamp_ms=0)

        # 显示结果
        cv2.imshow('MediaPipe', image)

        # 按 'q' 退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
