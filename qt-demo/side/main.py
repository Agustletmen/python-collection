import sys
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QGuiApplication
from PySide6.QtQml import QQmlApplicationEngine

if __name__ == "__main__": # 用于检查当前模块是否作为主程序运行
    app = QGuiApplication(sys.argv)

    engine = QQmlApplicationEngine()
    engine.load('main.qml')

    if not engine.rootObjects():
        sys.exit(-1)

    sys.exit(app.exec())
