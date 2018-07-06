#!usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author:    pfwu
@E-mail:    wpf0610@mail.ustc.edu.cn
@file:      Download.py
@software:  pycharm
Created on  2018/7/6 0006 15:49

"""

import os
os.makedirs('./img/', exist_ok=True)

IMAGE_URL = "https://morvanzhou.github.io/static/img/description/learning_step_flowchart.png"

# 方法1
from urllib.request import urlretrieve
urlretrieve(IMAGE_URL, './img/image1.png')

# 方法2
import requests
r = requests.get(IMAGE_URL)
with open('./img/image2.png', 'wb') as f:
    f.write(r.content)


# 方法3 -- 大文件用这种方法，逐步下载
r = requests.get(IMAGE_URL, stream=True)    # stream loading

with open('./img/image3.png', 'wb') as f:
    for chunk in r.iter_content(chunk_size=32):
        f.write(chunk)