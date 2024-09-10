from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import glfw
import ctypes

# 初始化 GLFW
if not glfw.init():
    raise Exception("glfw can not be initialized!")

# 创建窗口
win = glfw.create_window(500, 500, "My OpenGL window", None, None)

if not win:
    glfw.terminate()
    raise Exception("glfw window can not be created!")

# 设置当前上下文
glfw.make_context_current(win)

# 设置回调函数
glfw.set_window_close_callback(win, lambda x: glfw.set_window_should_close(x, False))

# 渲染循环
while not glfw.window_should_close(win):
    # 清除颜色缓冲区
    glClear(GL_COLOR_BUFFER_BIT)
    
    # 设置背景颜色
    glClearColor(0.1, 0.1, 0.1, 1)
    
    # 绘制三角形
    glBegin(GL_TRIANGLES)
    glColor3f(1.0, 0.0, 0.0)
    glVertex2f(-0.5, -0.5)
    glColor3f(0.0, 1.0, 0.0)
    glVertex2f(0.5, -0.5)
    glColor3f(0.0, 0.0, 1.0)
    glVertex2f(0.0, 0.5)
    glEnd()
    
    # 交换缓冲区
    glfw.swap_buffers(win)
    glfw.poll_events()

# 清理资源
glfw.terminate()