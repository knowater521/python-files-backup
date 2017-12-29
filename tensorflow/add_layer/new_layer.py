#!usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author:    pfwu
@file:      new_layer.py
@software:  pycharm
Created on  2017/12/29 0029 22:42

"""

import tensorflow as tf

def add_layer(inputs,in_size,out_size,activation_function=None):   #默认无激活函数
    Weights = tf.Variable(tf.random_normal([in_size,out_size]))
    biases = tf.Variable(tf.zeros([1,out_size])) + 0.1             #偏置初始值推荐不为0
    Wx_plus_b = tf.matmul(inputs,Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

