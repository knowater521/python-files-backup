#!usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author:    pfwu
@file:      Save.py
@software:  pycharm
Created on  2018/1/1 0001 22:34

"""

import tensorflow as tf
import numpy as np

# save to file
# remember to define the same dtype and shape when restore

# W = tf.Variable([[1,2,3],[3,4,5]],dtype=tf.float32,name='weights')
# b = tf.Variable([[1,2,3]],dtype=tf.float32,name='biases')
#
# init = tf.global_variables_initializer()
#
# saver = tf.train.Saver()
#
# with tf.Session() as sess:
#     sess.run(init)
#     save_path = saver.save(sess,'net_pra/save_net.ckpt')
#     print('Save to path:',save_path)


# restore variables
# redefine the same shape and dtype for your variables

W = tf.Variable(np.arange(6).reshape((2,3)),dtype=tf.float32,name='weights')
b = tf.Variable(np.arange(3).reshape((1,3)),dtype=tf.float32,name='biases')

# not need init step

saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess,'net_pra/save_net.ckpt')
    print('weights:',sess.run(W))
    print('biases:',sess.run(b))