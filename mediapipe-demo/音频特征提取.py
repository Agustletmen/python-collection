import wave
import numpy as np
import mediapipe as mp

# 初始化 MediaPipe 音频处理模块
mp_audio = mp.solutions.audio
audio_processing = mp_audio.AudioProcessing()

# 读取音频文件
audio_file = wave.open('example.wav', 'rb')
sample_rate = audio_file.getframerate()
n_frames = audio_file.getnframes()
audio_data = audio_file.readframes(n_frames)
audio_file.close()

# 将音频数据转换为 numpy 数组
audio_array = np.frombuffer(audio_data, dtype=np.int16)

# 处理音频数据
with audio_processing as audio_processor:
    audio_result = audio_processor.process(audio_array, sample_rate)

# 提取特征
if audio_result is not None:
    # 这里可以根据具体需求提取不同的特征，例如频谱特征
    spectrogram = audio_result.spectrogram
    print("提取的频谱特征形状:", spectrogram.shape)    