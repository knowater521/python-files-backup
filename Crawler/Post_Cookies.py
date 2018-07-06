#!usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author:    pfwu
@E-mail:    wpf0610@mail.ustc.edu.cn
@file:      Post_Cookies.py
@software:  pycharm
Created on  2018/7/6 0006 14:57

"""

import requests
import webbrowser

param = {"wd":"你好"}
response = requests.get('http://www.baidu.com',params=param)
# webbrowser.open(response.url)
print(response.url)   # http://www.baidu.com/?wd=%E4%BD%A0%E5%A5%BD 在?wd前缺少一个s


# http://pythonscraping.com/pages/files/form.html  这是一个测试爬虫的网址
# 在使用post时，填入的url并不是上述url，而是在该网页填入信息后submit进入的url
data = {'firstname':'pf','lastname':'wu'}
response = requests.post('http://pythonscraping.com/pages/files/processing.php',data=data) #
print(response.text)


# 上传一张图片  http://pythonscraping.com/files/form2.html
file = {'uploadFile': open('E:\python files\Crawler\99.png', 'rb')}
r = requests.post('http://pythonscraping.com/files/processing2.php', files=file)
print(r.text)


# 登录  http://pythonscraping.com/pages/cookies/login.html
payload = {'username':'pfwu','password':'password'}

r = requests.post('http://pythonscraping.com/pages/cookies/welcome.php',data=payload)
print(r.text)        # 登录出错，因为登录是连续的过程，python无法处理，故需要cookies
print(r.cookies.get_dict())

r = requests.get('http://pythonscraping.com/pages/cookies/profile.php',cookies=r.cookies)
print(r.ok)


# 使用Session ，不需要传入cookies
session = requests.Session()
r = session.post('http://pythonscraping.com/pages/cookies/welcome.php',data=payload)
print(r.cookies.get_dict())
r = session.get('http://pythonscraping.com/pages/cookies/profile.php')
print(r.status_code)


# 尝试登录百度

sess = requests.Session()
