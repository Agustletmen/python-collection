import cv2
import mediapipe as mp

"""
姿态检测可以识别身体的 33 个关键点，常用于运动分析、健身指导等场景。
"""
def main():
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False,  # Set to False for video processing
                        model_complexity=1,
                        smooth_landmarks=True,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5)
    drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)  # Use the default camera (index 0)

    while True:
        ret, frame = cap.read()  # Read the frame from the camera
        if not ret:
            print("Failed to grab frame")
            break

        # Convert the BGR image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False  # Performance optimization
        results = pose.process(image)

        # Draw the pose landmarks on the image
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.pose_landmarks:
            drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Display the image with landmarks
        cv2.imshow('Pose Detection', image)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()