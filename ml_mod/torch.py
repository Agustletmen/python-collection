"""
Torch，一个使用 Lua 编写的深度学习方面的库，底层是 C/C++
pytorch，torch库的Python版本
1. TorchVision：CV 领域的 “工具包”，封装 CV 常用数据集、模型、预处理函数
2. TorchText：NLP 领域专用库
  a. TorchData：新一代数据处理框架，替代 TorchText 的部分功能
3. TorchAudio：语音领域工具包
4. TorchScript：PyTorch 模型的 “中间表示”，连接训练与部署
5. TorchServe：PyTorch 官方模型服务框架，快速将模型封装为 HTTP/REST API；


pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
"""

"""
PyTorch Lightning 是 基于 PyTorch 的高层训练框架，核心目标是 “剥离 PyTorch 代码中的 boilerplate（重复模板代码）”
pip install pytorch-lightning
"""


def show():
    import torch
    print('CUDA版本:', torch.version.cuda)
    print('pytorch版本:', torch.__version__)
    print('显卡是否可用:', '可用' if (torch.cuda.is_available()) else '不可用')
    print('显卡数量:', torch.cuda.device_count())
    print('是否支持BF16数字格式:', '支持' if (torch.cuda.is_bf16_supported()) else '不支持')
    print('当前显卡型号:', torch.cuda.get_device_name())
    print('当前显卡的CUDA算力:', torch.cuda.get_device_capability())
    print('当前显卡的总显存:', torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024, 'GB')
    print('是否支持TensorCore:', '支持' if (torch.cuda.get_device_properties(0).major >= 7) else '不支持')
    print('当前显卡的显存使用率:', torch.cuda.memory_allocated(0) / torch.cuda.get_device_properties(0).total_memory * 100,'%')