# -*- coding:utf-8 -*-
import sys
from bs4 import BeautifulSoup  # 网页解析，获取数据
import re  # 正则表达式，进行文字匹配
import urllib.request  # 指定URL，获取网页数据
import urllib.response
import xlwt  # 进行excel操作

"""
爬取豆瓣Top250条电影的信息
爬虫的步骤：
    1、爬取网页
    2、解析数据
    3、保存数据
"""

# 创建正则表达式对象，表示解析规则（字符串的模式）
# 影片链接
findLink = re.compile(r'<a href="(.*?)">')  # <a href="https://movie.douban.com/subject/1291546/">

# 影片图片
findImgSrc = re.compile(r'<img.*src="(.*?)"', re.S)  # re.S 让换行符包含在字符中

# 影片片名
findTitle = re.compile(r'<span class="title">(.*?)</span>')
findTitle = re.compile(r'<span class="title">(.*?)</span>')

# 影片评分
findRating = re.compile(r'<span class="rating_num" property="v:average">(.*)</span>')

# 评价人数
findJudge = re.compile(r'<span>(\d*)人评价</span>')

# 概况
findInq = re.compile(r'<span class="inq">(.*)</span>')

# 影片相关内容
findBd = re.compile(r'<p class="">(.*?)</p>', re.S)  # 忽视换行符


def main():
    baseurl = "https://movie.douban.com/top250?start="  # 豆瓣的通用url

    datalist = getData(baseurl)  # 爬取网页

    savePath = ".\\豆瓣Top250.xls"
    saveData(datalist, savePath)  # 保存数据


# 获取html并从中解析所需的数据
def getData(baseurl):
    datalist = []
    for i in range(0, 10):  # 调用获取页面信息的函数，一页25条电影信息，10次即可获取top250条信息 左闭右开
        url = baseurl + str(i * 25)  # 0~250
        html = askURL(url)  # 保存获取到的网页源码
        # print(html)

        # （逐一）解析数据
        soup = BeautifulSoup(html, "html.parser")
        for item in soup.find_all('div', class_="item"):  # 查找符合要求的字符串，形成列表   （这里的class尾部加了_，是为了和关键字class区分）
            # print(item)
            data = []  # 保存一部电影的所有信息
            item = str(item)  # 把数据都转型为string

            # 从item中找寻 影片的链接
            link = re.findall(findLink, item)[0]  # re库通过正则表达式查找指定字符串
            data.append(link)  # 添加链接

            # 从item中找寻 影片的封面图片
            imgScr = re.findall(findImgSrc, item)[0]
            data.append(imgScr)  # 添加图片

            # 从item中找寻 影片的名字（可能含有多个）
            titles = re.findall(findTitle, item)  # 片名可能只有一个中文名，没有外文名
            if len(titles) == 2:
                ctitle = titles[0]  # 中文名
                data.append(ctitle)

                otitle = titles[1].replace(u'\xa0', u"")  # 外文名，去掉无关的符号
                otitle = otitle.replace("/", "")
                data.append(otitle)
            else:
                data.append(titles[0])
                data.append(' ')  # excel表格中，外文名留空

            # 从item中找寻 影片的评分
            rating = re.findall(findRating, item)[0]
            data.append(rating)  # 添加评分

            # 从item中找寻 影片的评价人数
            judgeNum = re.findall(findJudge, item)
            if len(judgeNum) != 0:
                judgeNum = judgeNum[0]
                data.append(judgeNum)  # 添加评价人数
            else:
                data.append(" ")  # 留空

            # 从item中找寻 影片的概述
            inq = re.findall(findInq, item)
            if len(inq) != 0:
                inq = inq[0].replace("。", "")  # 去掉句号
                data.append(inq)  # 添加概述
            else:
                data.append(" ")  # 留空

            # 从item中找寻 影片的相关内容
            bd = re.findall(findBd, item)[0]
            bd = re.sub('<br(\s+)?/>(\s+)?', "", bd)  # 去掉<br/>
            bd = re.sub('/', " ", bd)
            bd = re.sub('\xa0', '', bd)
            data.append(bd.strip())  # 去掉前后的空格

            datalist.append(data)  # 把处理好的一步电影信息放入
    # 打印存储好的信息
    for i in datalist:
        print(i)

        # print(link)
    return datalist


# 得到指定URL的网页内容
def askURL(url):
    # 获取数据步骤
    # 1、使用urllib.Request生成请求request
    # 2、使用urllib.urlopen发生请求获取响应response
    # 3、使用urllib.read获取页面的内容
    # 4、在访问页面是经常会出现错误，加入异常捕获语句

    # （用户代理）模拟浏览器头部信息，向服务器发送信息；告诉服务器，我们是什么类型的机器、浏览器（本质上是告诉浏览器，我们可以接受什么水平的文件内容）
    header = {  # 使用字典
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
                      'Chrome/103.0.0.0 Safari/537.36 '
    }
    html = ""

    request = urllib.request.Request(url, headers=header)  # 发送请求
    try:
        response = urllib.request.urlopen(request)  # 取得响应数据
        html = response.read().decode("utf-8")  # 从响应数据中获取网页内容
        # print(html)
    except urllib.error.URLError as e:
        if hasattr(e, "code"):  # hasattr(object, name) 用于判断对象是否包含对应的属性
            print(e.code)
        if hasattr(e, "reason"):
            print(e.reason)
    return html


# 保存数据
def saveData(dataList, savePath):
    book = xlwt.Workbook(encoding="utf-8", style_compression=0)  # 创建workbook对象
    sheet = book.add_sheet('豆瓣电影Top250', cell_overwrite_ok=True)  # 创建工作表
    col = ("电影详情链接", "图片链接", "影片中文名", "影片外文名", "评分", "评分数", "概况", "相关信息")
    for i in range(0, 8):
        sheet.write(0, i, col[i])  # 列名
    for i in range(0, 250):
        print("第%d条" % (i + 1))
        data = dataList[i]
        for j in range(0, 8):
            sheet.write(i + 1, j, data[j])  # 写入参数，（行，列，内容） 从0开始

    book.save(savePath)  # 保存数据表


if __name__ == "__main__":
    main()
