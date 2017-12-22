# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 20:37:32 2017

@author: pfwu
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

titanic = pd.read_csv('titanic.txt')

#特征的选择
x = titanic[['pclass','age','sex']]
y = titanic['survived']



  
#首先补充age里的数据，为了使影响最小，可以使用平均数或中位数

x['age'].fillna(x['age'].mean(),inplace=True)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=33)

vec = DictVectorizer(sparse = False)

#转换完成后，凡是类别型的特征都单独剥离出来，独成一列特征，数值型的则保留不变
x_train_transformed = vec.fit_transform(x_train.to_dict(orient='record'))
print(vec.feature_names_)

x_test_transformed = vec.transform(x_test.to_dict(orient='record'))

#使用单一的决策树
dtc = DecisionTreeClassifier()
dtc.fit(x_train_transformed,y_train)
dtc_y_predict = dtc.predict(x_test_transformed)

#使用随机森林分类器进行集成模型训练及预测

rfc = RandomForestClassifier()
rfc.fit(x_train_transformed,y_train)
rfc_y_predict = rfc.predict(x_test_transformed)

#使用提升方法进行集成模型的训练及预测
gbc = GradientBoostingClassifier()
gbc.fit(x_train_transformed,y_train)
gbc_y_predict = gbc.predict(x_test_transformed)

print('单一决策树的准确率:',dtc.score(x_test_transformed,y_test))
print(classification_report(dtc_y_predict,y_test))

print('随机森林的准确率:',rfc.score(x_test_transformed,y_test))
print(classification_report(rfc_y_predict,y_test))

print('提升方法的准确率:',gbc.score(x_test_transformed,y_test))
print(classification_report(gbc_y_predict,y_test))

