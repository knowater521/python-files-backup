#!usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author:    pfwu
@file:      beautifulsoup_basic.py
@software:  pycharm
Created on  2018/1/5 0005 18:22

"""

from bs4 import BeautifulSoup
from urllib.request import urlopen


# if has Chinese apply decode
html = urlopen("https://morvanzhou.github.io/static/scraping/basic-structure.html"
               ).read().decode('utf-8')
# print(html)

soup = BeautifulSoup(html,features='lxml')
print(soup.h1)
print(soup.p)

all_href = soup.find_all('a')
# for l in all_href:
#     print(l['href'])
href = [l ['href'] for l in all_href]
print(href)










