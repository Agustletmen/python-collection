"""
tensorflow
pip install tensorflow # Current stable release for CPU
pip install tensorflow[and-cuda] # Current stable release for GPU (Linux / WSL2)

Keras 原本是独立的高层 API，后被 TensorFlow 整合为官方子模块（tf.keras），与 TensorFlow 底层无缝衔接。
pip install keras # 不要运行这个！
"""
def show():
    import tensorflow as tf
    from tensorflow import keras

    # 输出版本
    print("TensorFlow 版本:", tf.__version__)
    print("Keras 版本:", keras.__version__)

    # 测试GPU（如果有）
    print("GPU 可用:", tf.config.list_physical_devices('GPU'))