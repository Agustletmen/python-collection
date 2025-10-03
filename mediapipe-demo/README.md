使用的是google自研的BlazeFace、BlazePose等模型

绘图
import mediapipe.python.solutions.drawing_styles
import mediapipe.python.solutions.drawing_utils

人脸边界框、6 个面部关键点（如双眼中心、鼻尖）及置信度。
import mediapipe.python.solutions.face_detection 

提供高精度人脸网格（面部关键点）检测模型，可检测 468 个精细的 3D 面部关键点
import mediapipe.python.solutions.face_mesh 
import mediapipe.python.solutions.face_mesh_connections 人脸网格关键点之间的连接关系

提供手部检测与关键点识别模型，可检测单 / 双手的 21 个 3D 关键点
import mediapipe.python.solutions.hands
import mediapipe.python.solutions.hands_connections

提供全身多模态姿态检测模型，整合了面部网格、手部关键点、身体姿态三种检测能力。
import mediapipe.python.solutions.holistic

提供单目 3D 物体检测模型，可从 2D 图像中估计常见物体（如杯子、椅子、鞋子）的 3D 边界框和空间姿态。
import mediapipe.python.solutions.objectron

提供人体姿态检测模型，专注于身体关键关节点（如头部、肩膀、手肘、膝盖等 33 个 3D 关键点）的检测与跟踪。
import mediapipe.python.solutions.pose

提供轻量级人像分割模型，快速分离图像中的前景（人体）和背景。
import mediapipe.python.solutions.selfie_segmentation