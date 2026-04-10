import cv2
import mediapipe

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

# 绘图
mp_drawing = mediapipe.tasks.vision.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# 人脸边界框、6 个面部关键点（如双眼中心、鼻尖）及置信度。
faceDetector = mediapipe.tasks.vision.FaceDetector.create_from_options(
    mediapipe.tasks.vision.FaceDetectorOptions(
        base_options=mediapipe.tasks.BaseOptions(model_asset_path=None),
        min_suppression_threshold=0.5  # 最小检测置信度
    )
)

# 提供高精度人脸网格（面部关键点）检测模型，可检测 468 个精细的 3D 面部关键点
faceLandmarker = mediapipe.tasks.vision.FaceLandmarker.create_from_options(
    mediapipe.tasks.vision.FaceLandmarkerOptions(
        base_options=mediapipe.tasks.BaseOptions(model_asset_path=None),
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
)

# 提供单目 3D 物体检测模型，可从 2D 图像中估计常见物体（如杯子、椅子、鞋子）的 3D 边界框和空间姿态。
objectDetector = mediapipe.tasks.vision.ObjectDetector.create_from_options(
    mediapipe.tasks.vision.ObjectDetectorOptions(
        base_options=mediapipe.tasks.BaseOptions(model_asset_path=None),
        max_results=5,
        score_threshold=0.5
    )
)

# 提供轻量级人像分割模型，快速分离图像中的前景（人体）和背景。
imageSegmenter = mediapipe.tasks.vision.ImageSegmenter.create_from_options(
    mediapipe.tasks.vision.ImageSegmenterOptions(
        base_options=mediapipe.tasks.BaseOptions(model_asset_path=None),
        output_category_mask=True,
        output_confidence_masks=False
    )
)

imageClassifier = mediapipe.tasks.vision.ImageClassifier.create_from_options(
    mediapipe.tasks.vision.ImageClassifierOptions(
        base_options=mediapipe.tasks.BaseOptions(model_asset_path=None),
        max_results=5,
        score_threshold=0.5
    )
)

interactiveSegmenter = mediapipe.tasks.vision.InteractiveSegmenter.create_from_options(
    mediapipe.tasks.vision.InteractiveSegmenterOptions(
        base_options=mediapipe.tasks.BaseOptions(model_asset_path=None),
        output_confidence_masks=True,
        output_category_mask=True
    )
)

imageEmbedder = mediapipe.tasks.vision.ImageEmbedder.create_from_options(
    mediapipe.tasks.vision.ImageEmbedderOptions(
        base_options=mediapipe.tasks.BaseOptions(model_asset_path=None)
    )
)

# 提供手部检测与关键点识别模型，可检测单 / 双手的 21 个 3D 关键点
handLandmarker = mediapipe.tasks.vision.HandLandmarker.create_from_options(
    mediapipe.tasks.vision.HandLandmarkerOptions(
        base_options=mediapipe.tasks.BaseOptions(model_asset_path=None),
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
)

# 提供人体姿态检测模型，专注于身体关键关节点（如头部、肩膀、手肘、膝盖等 33 个 3D 关键点）的检测与跟踪。
poseLandmarker = mediapipe.tasks.vision.PoseLandmarker.create_from_options(
    mediapipe.tasks.vision.PoseLandmarkerOptions(
        base_options=mediapipe.tasks.BaseOptions(model_asset_path=None),
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
)

# 提供全身多模态姿态检测模型，整合了面部网格、手部关键点、身体姿态三种检测能力。
holisticLandmarker = mediapipe.tasks.vision.HolisticLandmarker.create_from_options(
    mediapipe.tasks.vision.HolisticLandmarkerOptions(
        base_options=mediapipe.tasks.BaseOptions(model_asset_path=None),
        min_face_detection_confidence=0.5,
        min_pose_detection_confidence=0.5,
    )
)

gestureRecognizer = mediapipe.tasks.vision.GestureRecognizer.create_from_options(
    mediapipe.tasks.vision.GestureRecognizerOptions(
        base_options=mediapipe.tasks.BaseOptions(model_asset_path=None),
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
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

        # 显示结果
        cv2.imshow('MediaPipe Face Detection', image)

        # 按 'q' 退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
