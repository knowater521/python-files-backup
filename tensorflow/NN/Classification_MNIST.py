#!usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author:    pfwu
@file:      Classification_MNIST.py
@software:  pycharm
Created on  2017/12/31 0031 20:22

"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#有一些关于tensorboard的问题
# number 1 to 10 data
mnist = input_data.read_data_sets('..\MNIST_data', one_hot=True)


def add_layer(inputs, in_size, out_size, n_layer,activation_function=None,):
    # add one more layer and return the output of this layer
    layer_name = 'layer%s'%n_layer
    with tf.name_scope(layer_name):
        with tf.name_scope('Weights'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='W')
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='b')
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.matmul(inputs, Weights) + biases
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b, )
        return outputs


def compute_accuracy(v_xs,v_ys):
    global prediction
    y_pre = sess.run(prediction,feed_dict={xs:v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre,1),tf.argmax(v_ys,1))  #对比预测值是否等于真实值
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    result = sess.run(accuracy,feed_dict={xs:v_xs,ys:v_ys})
    return result


# define placeholder for inputs to network
with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32,[None,784],name='x_input')
    ys = tf.placeholder(tf.float32,[None,10],name = 'y_input')

# add output_layer
prediction = add_layer(xs,784,10,1,activation_function=tf.nn.softmax)
with tf.name_scope('corss_entropy'):
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                              reduction_indices=[1]))           #loss
    tf.summary.scalar('cross_entropy',cross_entropy)

with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.Session()
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter('logs/NN_MNIST',sess.graph)
sess.run(tf.global_variables_initializer())

for i in range(1000):
    batch_xs,batch_ys = mnist.train.next_batch(100)  #batch size 100
    sess.run(train_step,feed_dict={xs:batch_xs,ys:batch_ys})
    if i%50 == 0:
        print(compute_accuracy(mnist.test.images,mnist.test.labels))
        res = sess.run(merged,feed_dict={xs:batch_xs,ys:batch_ys})
        writer.add_summary(res,i)






