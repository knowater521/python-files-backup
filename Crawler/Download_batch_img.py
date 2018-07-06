#!usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author:    pfwu
@E-mail:    wpf0610@mail.ustc.edu.cn
@file:      Download_batch_img.py
@software:  pycharm
Created on  2018/7/6 0006 16:01

"""

# 在国家地理网站批量下载图片，首先用正则表达式找到图片的url，再进行下载  （失败了，好像是因为这个网站变成动态了）

from bs4 import BeautifulSoup
import requests

URL = "http://www.nationalgeographic.com.cn/animals/"

# 用beautifulsoup找到带有 img_list的 <ul>
html = requests.get(URL).text
soup = BeautifulSoup(html, 'lxml')
img_ul = soup.find_all('ul', {"class": "img_list"})

# 从 ul 中找到所有的 <img>, 然后提取 <img> 的 src 属性
# 接着, 就用之前在 requests 下载那节内容里提到的一段段下载.

for ul in img_ul:
    imgs = ul.find_all('img')
    for img in imgs:
        url = img['src']
        r = requests.get(url, stream=True)
        image_name = url.split('/')[-1]
        with open('./img/%s' % image_name, 'wb') as f:
            for chunk in r.iter_content(chunk_size=128):
                f.write(chunk)
        print('Saved %s' % image_name)