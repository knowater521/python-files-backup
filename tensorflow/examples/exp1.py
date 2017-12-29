# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 20:52:57 2017

@author: pfwu
"""
#!usr/bin/env python


import tensorflow as tf
import numpy as np

#create data
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3

#create tensorflow structure

Weights = tf.Variable(tf.random_uniform([1],-1.0,1.0))   # 1 dimension range: -1 ~ 1
biases = tf.Variable(tf.zeros([1]))                       # 1 dimension init values 0

y = Weights * x_data + biases

loss = tf.reduce_mean(tf.square(y-y_data))

optimizer = tf.train.GradientDescentOptimizer(0.5) #learning rate 0.5
train = optimizer.minimize(loss)


#init the net
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)   # activate init

for step in range(201):
    sess.run(train)
    if step%20 == 0:
        print(step,sess.run(Weights),sess.run(biases))


