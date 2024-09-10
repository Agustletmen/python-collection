import json
import math

import requests
import subprocess  # 执行cmd命令
import os
from bs4 import BeautifulSoup

from tqdm import tqdm  # 进度条显示

directory = "./bilibili"  # 保存视频的文件夹名称


# # 调用关系：main --> make_mp4 --> get_video_and_audio --> get_url
# 1、传入url，解析地址
# 2、下载视频的【视频流】和【音频流】
# 3、使用ffmpeg工具合成视频，删除临时文件


def main():
    if not os.path.exists(f'./{directory}'):
        os.mkdir(f'./{directory}')
    url = input('请输入视频的B站地址：')
    make_mp4(url)
    print('视频下载完毕')


# 第一步：解析网页元素，返回视频内容的下载地址
def get_url(url):
    print('地址解析中')
    video_html_response = requests.get(url)  # 取得响应数据：requests.models.Response
    video_html_str = video_html_response.text  # 将响应数据转化为 str

    # 使用了 BeautifulSoup 的 lxml Html 解析器
    text = BeautifulSoup(video_html_str, features='lxml')

    title = text.find('title').contents[0].replace(' ', ',').replace('/', ',')
    items = text.find_all('script')[2]
    items = items.contents[0].replace('window.__playinfo__=', '')

    # 将json类型转换为字典类型
    obj = json.loads(items)
    video_url = obj["data"]["dash"]["video"][0]["baseUrl"]
    audio_url = obj["data"]["dash"]["audio"][0]["baseUrl"]
    print('地址解析完成')
    return video_url, audio_url, title


# 第二步：下载视频内容（视频流、音频流）
def get_video_and_audio(url):
    print('开始发送请求')

    urls = get_url(url)

    video_url = urls[0]
    audio_url = urls[1]
    title = urls[2]

    print('请求成功返回')

    # print('开始下载视频')
    print(f'./{directory}/video.mp4')

    # 用户代理请求头
    headers = {
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) '
                      'Chrome/101.0.4951.54 Safari/537.36',
        # 'Range': 'bytes=0-29609553',
        'Referer': url
    }

    # 下载MP4
    with open(f'./{directory}/video.mp4', 'wb') as video:
        # video.write(requests.get(video_url, headers=headers).content)

        res = requests.get(video_url, headers=headers, stream=True)
        content_size = math.ceil(int(res.headers['Content-Length']) / 1024)
        for data in tqdm(iterable=res.iter_content(1024), total=content_size, unit='k', desc='mp4下载中'):
            video.write(data)

    # 下载MP3
    with open(f'./{directory}/audio.mp3', 'wb') as audio:
        # audio.write(requests.get(audio_url, headers=headers).content)
        res = requests.get(audio_url, headers=headers, stream=True)
        content_size = math.ceil(int(res.headers['Content-Length']) / 1024)
        for data in tqdm(iterable=res.iter_content(1024), total=content_size, unit='k', desc='mp3下载中'):
            audio.write(data)

    # print('下载完毕')

    return f'./{directory}/video.mp4', f'./{directory}/audio.mp3', title


# 第三步：合成视频，并删除数据文件
def make_mp4(url):
    # 获取 mp3、mp4 和 title
    file = get_video_and_audio(url)
    mp4_file = file[0]
    mp3_file = file[1]
    title = file[2]

    print("mp4_file = ", mp4_file)
    print('mp3_file = ', mp3_file)
    print("title = ", title)

    # 使用 FFmpeg 对视频进行合成
    print('开始合成音频视频')
    cmd = f'ffmpeg -i {mp4_file} -i {mp3_file} -acodec copy -vcodec copy ./{directory}/{title}.mp4'  # ffmpeg操作指令
    subprocess.call(cmd, shell=True)  # 调用 cmd 执行命令
    print('合成完毕')

    # 善后工作，处理音频文件和视频文件
    print('正在删除临时文件')
    os.remove(mp4_file)
    os.remove(mp3_file)
    print('删除完毕')


if __name__ == '__main__':
    main()
