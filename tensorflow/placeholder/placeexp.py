#!usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author:   pfwu

@file:     placeexp.py

@software: pycharm

Created on 2017/12/29 0029 22:23

"""

import tensorflow as tf

input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

output = tf.multiply(input1,input2)

num1 = tf.placeholder(tf.float32)
num2 = tf.placeholder(tf.float32)

num3 = tf.div(num1,num2)

with tf.Session() as sess:
    print(sess.run(output,feed_dict={input1:[7.],input2:[2.]}))  #用feed_dict 传值进place_holder
    print(sess.run(num3,feed_dict={num1:[25],num2:[5]}))



