import cv2

# 读取图像
image = cv2.imread('example.jpg', 0)  # 以灰度模式读取图像

# 检查图像是否成功读取
if image is not None:
    # 应用 Canny 边缘检测
    edges = cv2.Canny(image, 100, 200)

    # 显示原始图像和边缘检测结果
    cv2.imshow('Original Image', image)
    cv2.imshow('Edges', edges)

    # 等待按键事件
    cv2.waitKey(0)
    # 关闭所有窗口
    cv2.destroyAllWindows()
else:
    print("无法读取图像，请检查文件路径。")