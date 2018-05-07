#!usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author:    pfwu
@E-mail:    wpf0610@mail.ustc.edu.cn
@file:      02_WithPara.py
@software:  pycharm
Created on  2018/5/7 0007 9:48

"""


import logging




def dec(func):

    def wrapper(*args,**kwargs):    # *args 表示可变参数  **kwargs 表示关键字参数
        logging.basicConfig(level=logging.DEBUG, format='%(pathname)s - %(lineno)s - %(asctime)s - %(name)s - %(levelname)s - %(message)s',
                            filename='log1.txt',filemode='a+')
        logging.warning("%s is running"%func.__name__)
        logging.debug("This is debug information")
        return func(*args,**kwargs)

    return wrapper

def double_dec(func):

    def wrapper(*args,**kwargs):    # *args 表示可变参数  **kwargs 表示关键字参数

        return func(*args,**kwargs)*2

    return wrapper

@dec   # 加此语句可避免 bar = dec(bar)等的赋值
@double_dec

def bar(number):
    return number*3



if __name__ == '__main__':
    print(bar(2))