# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 15:38:20 2017

@author: Administrator
"""

from sklearn import tree
#X = [[0,0],[2,2]]
#y=[0.5,2.5]
#clf = tree.DecisionTreeRegressor()
#clf = clf.fit(X,y)
#print(clf.predict([[1,1]]))
print(__doc__)
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
rng = np.random.RandomState(1)
X = np.sort(5*rng.rand(80,1),axis=0)
y = np.sin(X).ravel()  #变成行向量
y[::5] += 3*(0.5 - rng.rand(16))


#
#regr_1 = DecisionTreeRegressor(max_depth = 2)
#regr_2 = DecisionTreeRegressor(max_depth = 5)
#
#regr_1.fit(X,y)
#regr_2.fit(X,y)
#
#
#
#X_test = np.arange(0.0,5.0,0.01)[:,np.newaxis]  #np.newaxis equal to None, add an axis for X_test
#y_1 = regr_1.predict(X_test)
#y_2 = regr_2.predict(X_test)
#
#plt.figure()
#plt.scatter(X,y,s=20,edgecolor='black',c='darkorange',label='data')
#plt.plot(X_test,y_1,color='cornflowerblue',label='max_depth=2',linewidth=2)
#plt.plot(X_test,y_2,color='yellowgreen',label='max_depth=5',linewidth=2)
#plt.xlabel("data")
#plt.ylabel('target')
#plt.title('Decision Tree Regression')
#plt.legend()
#plt.show()
