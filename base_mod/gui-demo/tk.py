import tkinter as tk
from tkinter import ttk # Tkinter 内置美化组件
from tkinter import messagebox

# 创建主窗口
root = tk.Tk()
root.title("Tkinter 入门示例")
root.geometry("300x200")  # 窗口大小（宽x高）

# 定义按钮点击事件
def on_click():
    messagebox.showinfo("提示", "按钮被点击啦！")

# 添加组件（标签 + 按钮）
label = tk.Label(root, text="Hello Tkinter!")
label.pack(pady=20)  # 布局（垂直间距20）

btn = ttk.Button(root, text="美化按钮", command=on_click)
# btn = tk.Button(root, text="点击我", command=on_click, width=15, height=2)
btn.pack()

# 启动主循环（监听用户操作）
root.mainloop()