import dlib
import cv2

# 打开视频文件
cap = cv2.VideoCapture('example.mp4')

# 读取第一帧
ret, frame = cap.read()
if not ret:
    print("无法读取视频帧")
    exit()

# 选择跟踪区域
x, y, w, h = cv2.selectROI(frame, False)
tracker = dlib.correlation_tracker()
rect = dlib.rectangle(x, y, x + w, y + h)
tracker.start_track(frame, rect)

while True:
    # 读取下一帧
    ret, frame = cap.read()
    if not ret:
        break

    # 更新跟踪器
    tracker.update(frame)
    pos = tracker.get_position()
    x = int(pos.left())
    y = int(pos.top())
    w = int(pos.width())
    h = int(pos.height())

    # 绘制跟踪框
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # 显示结果
    cv2.imshow('Object Tracking', frame)

    # 按 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()