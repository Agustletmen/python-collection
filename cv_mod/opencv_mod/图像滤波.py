import cv2

# 读取图像
image = cv2.imread('example.jpg')

# 检查图像是否成功读取
if image is not None:
    # 应用高斯滤波
    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    # 显示原始图像和滤波后的图像
    cv2.imshow('Original Image', image)
    cv2.imshow('Blurred Image', blurred)

    # 等待按键事件
    cv2.waitKey(0)
    # 关闭所有窗口
    cv2.destroyAllWindows()
else:
    print("无法读取图像，请检查文件路径。")