import requests
from bs4 import BeautifulSoup
import os

# https://www.zrfsxs.com/txt/?id=39&p=1

# 请求页面并解析 HTML
# url = "https://www.zrfsxs.com"
for i in range(116):
    url = "https://www.zrfsxs.com/xiaoshuo/" + str(i + 1) + "/txt.html#dir"
    print(url)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.150 Safari/537.36"
    }
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")
    print("===============解析网页内容==============")
    print(soup)

    # 找到 id 为 list 的 div 元素
    list_div = soup.find("div", {"id": "list"})
    print("===============可获取list==============")
    print(list_div)

    # 找到所有的链接标签，并遍历处理
    for link in list_div.find_all("a"):
        href = link.get("href")  # 获取链接的 href 属性值
        file_name = link.string

        # 拼接完整的下载链接
        download_url = "https://www.zrfsxs.com" + href
        print("下载地址：" + download_url)

        # 下载文件
        response = requests.get(download_url)
        print(response.status_code)
        if response.status_code == 200:
            with open(
                "D:\\Code\\my-python\\Spider\\spider\\novel\\" + file_name, "wb"
            ) as f:
                f.write(response.content)
            print("ok")
        else:
            print("fail")
    # https://www.zrfsxs.com/txt/?id=39&p=1
