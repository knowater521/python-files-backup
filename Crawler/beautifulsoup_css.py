#!usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author:    pfwu
@file:      beautifulsoup_css.py
@software:  pycharm
Created on  2018/1/5 0005 19:21

"""

from bs4 import BeautifulSoup
from urllib.request import urlopen

html = urlopen("https://morvanzhou.github.io/static/scraping/list.html").read().decode('utf-8')
print(html)
soup = BeautifulSoup(html,features='lxml')
month = soup.find_all('li',{"class":"month"})
for m in month:
    print(m.get_text())  # print(m) <li class="month">一月</li>

jan = soup.find('ul',{"class":'jan'})
for d in jan.find_all('li'):
    print(d.get_text())




















