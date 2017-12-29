#!usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: pfwu
"""

import tensorflow as tf


state = tf.Variable(0,name='counter')
print(state.name)
print(state.value())

one = tf.constant(1)
new_value = tf.add(state,1)
update = tf.assign(state,new_value)   #把new_value赋给state

init = tf.global_variables_initializer()    #如果使用变量这一步尤其重要

with tf.Session() as sess:
    sess.run(init)
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))