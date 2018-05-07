#!usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author:    pfwu
@file:      Regressor.py
@software:  pycharm
Created on  2018/3/2 0002 15:52

"""


# 利用keras搭建一个回归神经网络
import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt # 可视化模块


# create some data
X = np.linspace(-1, 1, 200)
np.random.shuffle(X)    # randomize the data
Y = 0.5 * X + 2 + np.random.normal(0, 0.05, (200, ))
# plot data
plt.scatter(X, Y)


X_train, Y_train = X[:160], Y[:160]     # train 前 160 data points
X_test, Y_test = X[160:], Y[160:]       # test 后 40 data points


# bulid a neural network from the 1st layer to last layer

model = Sequential()   # 逐层添加的为网络
model.add(Dense(units=1,input_dim=1))  # 添加全连接层

# choose loss function and optimizing method
model.compile(loss='mse',optimizer='sgd')


# training
print('Training -----------')
for step in range(301):
    cost = model.train_on_batch(X_train, Y_train)
    if step % 100 == 0:
        print('train cost: ', cost)

# test
print('\nTesting ------------')
cost = model.evaluate(X_test, Y_test, batch_size=40)
print('test cost:', cost)
W, b = model.layers[0].get_weights()
print('Weights=', W, '\nbiases=', b)


# plotting the prediction
Y_pred = model.predict(X_test)
plt.scatter(X_test, Y_test)
plt.plot(X_test, Y_pred)
plt.show()










