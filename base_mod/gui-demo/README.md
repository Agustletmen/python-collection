# 桌面开发
Tkinter：自带，功能少
PySimpleGUI：底层封装了 Tkinter、PyQt、WxPython 等原生 GUI 库。
pip install pysimplegui

## Qt
PyQt
Pyside6（对应 Qt6）
Pyside2（对应 Qt5）

pip install pyside6
pip install pyside6-webengine

pyside6-designer：可视化拖拽设计界面，生成 .ui 文件
pyside6-uic：将.ui文件转换为 Python 代码
pyside6-uic -o <out.py> <in.ui>

## Kivy
跨平台能力极强（支持 Windows/Mac/Linux/Android/iOS）、支持多点触控、适合开发多媒体 / 游戏类 GUI；

pip install pyinstaller