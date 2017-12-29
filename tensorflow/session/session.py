# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 21:18:17 2017

@author: pfwu
"""
#!usr/bin/env python

import tensorflow as tf
import numpy as np

matrix1 = tf.constant([[3,3]])
matrix2 = tf.constant([[4],[2]])

product = tf.matmul(matrix1,matrix2)

##method 1
#sess = tf.Session()
#result = sess.run(product)
#print(result)
#sess.close()

# method2
with tf.Session() as sess:  #以sess打开Session,不用管Session是否关闭
    result2 = sess.run(product)
    print(result2)
