from PySide6.QtWidgets import (QApplication, QMainWindow, QLabel, 
                               QLineEdit, QPushButton, QVBoxLayout, QWidget)
import sys

# 定义主窗口类（继承 QMainWindow）
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.title = "PySide6 入门示例"
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(100, 100, 400, 300)  # 窗口位置（x,y）和大小（宽,高）

        # 创建中心部件和布局（垂直布局）
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # 添加组件
        self.label = QLabel("请输入内容：")
        layout.addWidget(self.label)

        self.input = QLineEdit()  # 输入框
        layout.addWidget(self.input)

        self.btn = QPushButton("提交")
        self.btn.clicked.connect(self.on_submit)  # 绑定点击事件（信号槽）
        layout.addWidget(self.btn)

    # 按钮点击事件处理
    def on_submit(self):
        text = self.input.text()
        if text:
            self.label.setText(f"你输入的是：{text}")
        else:
            self.label.setText("请输入有效内容！")

# 程序入口
if __name__ == "__main__":
    app = QApplication(sys.argv)  # 创建应用实例
    window = MainWindow()
    window.show()  # 显示窗口
    sys.exit(app.exec())  # 启动应用循环