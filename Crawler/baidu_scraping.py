#!usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author:    pfwu
@file:      baidu_scraping.py
@software:  pycharm
Created on  2018/1/5 0005 19:43

"""

from bs4 import BeautifulSoup
from urllib.request import urlopen
import re
import random

base_url = "https://baike.baidu.com"
his = ["/item/%E7%BD%91%E7%BB%9C%E7%88%AC%E8%99%AB/5162711"]
url = base_url + his[-1]   # his[-1] the last in his
html = urlopen(url).read().decode('utf-8')
soup = BeautifulSoup(html, features='lxml')
print(soup.find('h1').get_text(), '    url: ', url) # url:his[-1]


