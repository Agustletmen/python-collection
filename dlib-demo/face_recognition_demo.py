import face_recognition
import cv2
import os

# 用于存储已保存人物的人脸编码
known_face_encodings = []


def save_frame_image(frame):
    """
    保存整帧图像到当前位置
    :param frame: 整帧图像（OpenCV 格式）
    """
    # 生成文件名
    file_count = len([name for name in os.listdir('.') if os.path.isfile(name) and name.startswith('frame_')])
    file_name = f'frame_{file_count + 1}.jpg'
    # 保存图像
    cv2.imwrite(file_name, frame)
    print(f"整帧图像已保存为 {file_name}")


def detect_and_send():
    global known_face_encodings
    # 打开摄像头
    video_capture = cv2.VideoCapture(0)

    while True:
        # 读取一帧视频
        ret, frame = video_capture.read()
        if not ret:
            print("无法读取视频帧，退出循环。")
            break

        # 将图像从 BGR 颜色（OpenCV 使用）转换为 RGB 颜色（face_recognition 使用）
        rgb_frame = frame[:, :, ::-1]

        # 检测人脸位置
        face_locations = face_recognition.face_locations(rgb_frame)
        # 检测人脸地标
        face_landmarks_list = face_recognition.face_landmarks(rgb_frame, face_locations)

        new_person_detected = False

        for face_landmarks, (top, right, bottom, left) in zip(face_landmarks_list, face_locations):
            # 绘制矩形框
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            # 提取人脸编码，直接传入人脸地标信息
            face_encoding = face_recognition.face_encodings(rgb_frame)[0]

            # 检查是否为新人物
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            if not any(matches):
                new_person_detected = True
                known_face_encodings.append(face_encoding)

        if new_person_detected:
            # 保存整帧图像
            save_frame_image(frame)

        # 显示结果图像
        cv2.imshow('Video', frame)

        # 按 'q' 键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放摄像头并关闭窗口
    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    detect_and_send()
