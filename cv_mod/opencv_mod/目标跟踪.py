import cv2

tracker = cv2.TrackerCSRT_create()
tracking = False
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break  # 摄像头读取失败则退出

    # 等待按键（1毫秒超时）
    key = cv2.waitKey(1)
    
    # 按 'a' 键选择目标区域并开始跟踪
    if key == ord('a'):
        # 选择ROI（返回 (x, y, w, h)，若取消选择则返回 (0,0,0,0)）
        roi = cv2.selectROI('frame', frame, False)  # False 表示不显示交叉线
        if roi[2] > 0 and roi[3] > 0:  # 确保ROI有效（宽高>0）
            tracking = True
            tracker.init(frame, roi)
        else:
            tracking = False  # 取消选择则停止跟踪

    # 若正在跟踪，更新并绘制边界框
    if tracking:
        success, box = tracker.update(frame)
        if success:
            x, y, w, h = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # 显示帧（修正拼写错误：imgshow → imshow）
    cv2.imshow('frame', frame)

    # 按 'q' 键退出
    if key == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()