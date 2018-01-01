#!usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author:    pfwu
@file:      CNN.py
@software:  pycharm
Created on  2017/12/31 0031 22:28

"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 问题：各层有一些变量的名称没有传进去
# number 1 to 10 data
mnist = input_data.read_data_sets('..\MNIST_data', one_hot=True)


def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result


def weight_variable(shape,Wname):
    initial = tf.truncated_normal(shape, stddev=0.1,name=Wname)
    return tf.Variable(initial)


def bias_variable(shape,bname):
    initial = tf.constant(0.1, shape=shape,name=bname)
    return tf.Variable(initial)


def conv2d(x, W):
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x,pool_name):
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME',name=pool_name)

# define placeholder for inputs to network

with tf.name_scope('input_data'):
    xs = tf.placeholder(tf.float32, [None, 784],name='x_input')/255.   # 28x28
    ys = tf.placeholder(tf.float32, [None, 10],name='y_input')
    keep_prob = tf.placeholder(tf.float32,name='keep_prob')
    tf.summary.scalar('keep_prob',keep_prob)
    x_image = tf.reshape(xs, [-1, 28, 28, 1])
# print(x_image.shape)  # [n_samples, 28,28,1]

## conv1 layer ##

with tf.name_scope('conv1'):
    W_conv1 = weight_variable([5,5, 1,32],Wname='W_conv1') # patch 5x5, in size 1, out size 32
    b_conv1 = bias_variable([32],bname='b_conv1')
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1,name='conv1_out') # output size 28x28x32
    h_pool1 = max_pool_2x2(h_conv1,pool_name='pool1_out')                                         # output size 14x14x32

## conv2 layer ##

with tf.name_scope('conv2'):
    W_conv2 = weight_variable([5,5, 32, 64],Wname='W_conv2') # patch 5x5, in size 32, out size 64
    b_conv2 = bias_variable([64],bname='b_conv2')
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2,name='conv2_out') # output size 14x14x64
    h_pool2 = max_pool_2x2(h_conv2,pool_name='pool2_out')                                         # output size 7x7x64

## fc1 layer ##
with tf.name_scope('fc1'):
    W_fc1 = weight_variable([7*7*64, 1024],Wname='W_fc1')
    b_fc1 = bias_variable([1024],bname='b_fc1')
# [n_samples, 7, 7, 64] ->> [n_samples, 7*7*64]
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob,name='fc1_out')

## fc2 layer ##
with tf.name_scope('fc2'):
    W_fc2 = weight_variable([1024, 10],Wname='W_fc2')
    b_fc2 = bias_variable([10],bname='b_fc2')
    prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2,name='prediction')


# the error between prediction and real data
with tf.name_scope('cross_entropy'):
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                                  reduction_indices=[1]),name='cross_entropy')       # loss
    tf.summary.scalar('cross_entropy',cross_entropy)
with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

sess = tf.Session()
merged = tf.summary.merge_all()
init = tf.global_variables_initializer()
sess.run(init)
writer = tf.summary.FileWriter('logs',sess.graph)

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
    if i % 50 == 0:
        result = sess.run(merged,feed_dict={xs:batch_xs,ys:batch_ys,keep_prob:0.5})
        writer.add_summary(result,i)
        print(compute_accuracy(
            mnist.test.images[:1000], mnist.test.labels[:1000]))
