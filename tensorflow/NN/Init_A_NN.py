#!usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author:    pfwu
@file:      Init_A_NN.py
@software:  pycharm
Created on  2017/12/29 0029 22:49

"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def add_layer(inputs,in_size,out_size,activation_function=None):   #默认无激活函数
    Weights = tf.Variable(tf.random_normal([in_size,out_size]))
    biases = tf.Variable(tf.zeros([1,out_size])) + 0.1             #偏置初始值推荐不为0
    Wx_plus_b = tf.matmul(inputs,Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs


x_data = np.linspace(-1,1,300)[:,np.newaxis]  #?????
np.random.seed(1)
noise = np.random.normal(0,0.05,x_data.shape)
y_data = np.square(x_data) - 0.5


xs = tf.placeholder(tf.float32,[None,1]) #None 无论给多少个例子都行，后面的1是x的维度
ys = tf.placeholder(tf.float32,[None,1])

hidden_layer_out = add_layer(xs,1,10,activation_function=tf.nn.relu)  #隐藏层10个神经元
prediction = add_layer(hidden_layer_out,10,1,activation_function=None)

loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),reduction_indices=[1])) #求和后求平均

train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data,y_data)
plt.ion()    #加上这个之后，相当于holdon


for i in range(1000):
    sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
    if i%50 == 0:
        print(sess.run(loss,feed_dict={xs:x_data,ys:y_data}))
        try:
            ax.lines.remove(lines[0])  # 如果要求连续效果，必须抹除上次的线
        except Exception as e:
            pass
        prediction_value = sess.run(prediction,feed_dict={xs:x_data})
        lines = ax.plot(x_data,prediction_value,'r-',lw=5)
        plt.pause(0.1)


plt.show()

