#!usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author:    pfwu
@E-mail:    wpf0610@mail.ustc.edu.cn
@file:      Parameter.py
@software:  pycharm
Created on  2018/5/7 0007 10:07

"""


# 可变参数和关键字参数

def dec(fun):
    def wrapper(*args,**kwargs):
        print('%s is runing'%fun.__name__)
        return fun(*args,**kwargs)
    return wrapper


def calc(*numbers):   # 使用*numbers的方法定义函数参数，则函数参数的个数可变
    sum = 0
    for n in numbers:
        sum = sum + n*n
    return sum


@dec
def variable_parameter():

    print(calc(1,2,3))
    print(calc(1,2,3,4,5))  # 这两种没问题

    # print(calc([1,2,3])) # 直接传入List和tuple不行
    # print(calc((1,2,3)))

    print(calc(*[1,2,3,4])) # 在前面加*号就可以了
    print(calc(*(1,2,3)))


if __name__ == '__main__':
    variable_parameter()