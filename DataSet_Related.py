# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 20:03:00 2017

@author: Administrator
"""

import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
# load the CSV file as a numpy matrix
dataset = np.loadtxt('01.txt', delimiter=",")
# separate the data from the target attributes
x = dataset[:,0:7]
y = dataset[:,8]
#from sklearn import preprocessing 
##normalize the data attributes
#normalized_x = preprocessing.normalize(x)  #规格化
##standardize the data attributes
#standardized_x = preprocessing.scale(x)
#


#from sklearn import metrics
#from sklearn.ensemble import ExtraTreesClassifier
#model = ExtraTreesClassifier()#树方法，计算特征的信息量
#model.fit(x,y)
##display the relative importance of each attribute
#print(model.feature_importances_)
#
#from sklearn.feature_selection import RFE
#from sklearn.linear_model import LogisticRegression
#model = LogisticRegression()
##create the RFE model and select 3 attributes
#rfe = RFE(model,3)
#rfe = rfe.fit(x,y)
##summarize the selection of the attributes
#print(rfe.support_)  #看选取了哪几维特征
#print(rfe.ranking_)
#


##逻辑回归
#from sklearn.linear_model import LogisticRegression
#model = LogisticRegression()
#model.fit(x,y)
#print(model)
##make predictions
#expected = y
#predicted = model.predict(x)
##X=np.arange(1,769)
##pl.plot(X,y,'o')
##pl.show()
##summarize the fit of the model
#print(metrics.classification_report(expected,predicted))
#print(metrics.confusion_matrix(expected,predicted))

#朴素贝叶斯，主要任务是恢复训练样本的数据分布密度
#from sklearn import metrics
#from sklearn.naive_bayes import GaussianNB
#model = GaussianNB()
#model.fit(x,y)
#print(model)
##make predictions
#expected = y
#predicted = model.predict(x)
##summarize the fit of the model
#print(metrics.classification_report(expected,predicted))
#print(metrics.confusion_matrix(expected,predicted))

#k近邻
#from sklearn import metrics
#from sklearn.neighbors import KNeighborsClassifier
##fit a k-nearest neighbor model to the data
#model = KNeighborsClassifier()
#model.fit(x,y)
#print(model)
##make predictions
#expected = y
#predicted = model.predict(x)
##summarize the fit of the model
#print(metrics.classification_report(expected,predicted))
#print(metrics.confusion_matrix(expected,predicted))


#决策树
#from sklearn import metrics
#from sklearn.tree import DecisionTreeClassifier
##fit a CART model to the data
#model = DecisionTreeClassifier()
#model.fit(x,y)
#print(model)
##make predictions
#expected = y
#predicted = model.predict(x)
###summarize the fit of the model
#print(metrics.classification_report(expected,predicted))
#print(metrics.confusion_matrix(expected,predicted))

#SVM
#from sklearn import metrics
#from sklearn.svm import SVC
#model = SVC()
#model.fit(x,y)
#print(model)
#expected = y
#predicted = model.predict(x)
###summarize the fit of the model
#print(metrics.classification_report(expected,predicted))
#print(metrics.confusion_matrix(expected,predicted))
#

#
#import numpy as np
#from sklearn.linear_model import Ridge
#from sklearn.grid_search import GridSearchCV
##prepare a range of alpha values to test
#alphas = np.array([1,0.1,0.01,0.001,0.0001,0])
##create and fit a ridge regression model,testing each alpha
#model = Ridge()
#grid = GridSearchCV(estimator = model,param_grid=dict(alpha=alphas))
#grid.fit(x,y)
#print(grid)
##summarize the result of the grid search
#print(grid.best_score_)
#print(grid.best_estimator_.alpha)
#



import numpy as np
from scipy.stats import uniform as sp_rand
from sklearn.linear_model import Ridge
from sklearn.grid_search import RandomizedSearchCV
# prepare a uniform distribution to sample for the alpha parameter
param_grid = {'alpha': sp_rand()}
# create and fit a ridge regression model, testing random alpha values
model = Ridge()
rsearch = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=100)
rsearch.fit(x, y)
print(rsearch)
# summarize the results of the random parameter search
print(rsearch.best_score_)
print(rsearch.best_estimator_.alpha)


































