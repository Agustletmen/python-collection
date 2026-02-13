import dlib
import cv2

# 加载人脸检测器和关键点检测器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# 读取图像
image = cv2.imread('example.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 检测人脸
faces = detector(gray)

# 遍历检测到的人脸
for face in faces:
    # 检测关键点
    shape = predictor(gray, face)
    # 遍历关键点
    for i in range(0, 68):
        x = shape.part(i).x
        y = shape.part(i).y
        cv2.circle(image, (x, y), 2, (0, 0, 255), -1)

# 显示结果
cv2.imshow('Face Landmarks', image)
cv2.waitKey(0)
cv2.destroyAllWindows()