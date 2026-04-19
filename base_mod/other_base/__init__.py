"""
pybind11
"""


"""
setuptools：原生 distutils 的增强版，负责定义项目元数据、处理依赖、构建源码、打包分发


# 源代码分发（Source Distribution，sdist）
python setup.py sdist

# Egg 分发（Egg Distribution，旧格式）
python setup.py bdist_egg

# Wheel 二进制分发（Wheel Distribution）
pip install wheel
python setup.py bdist_wheel
"""


"""
wheel
Python 的二进制包格式（后缀为 .whl），同时也是生成 / 安装这种格式包的工具。替代传统的 .egg 格式
.whl（Wheel）是 Python 官方推荐的二进制分发格式（PEP 427 标准），本质是一个 zip 压缩包，包含预编译的代码（如 .py、.pyd、.so）和元数据（METADATA、RECORD 等）。

.pyd 是 Windows 系统下 Python 扩展模块的二进制文件（类似 Linux 的 .so 或 macOS 的 .dylib），本质是动态链接库（DLL），但专为 Python 解释器设计。
"""

"""
packaging
Python 打包相关的基础工具库，提供版本解析、依赖规范、包名验证等底层功能。
"""