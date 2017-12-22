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

titanic = pd.read_csv('titanic.txt')

#print(titanic.head())
print(titanic.info())  #数据是不完整的

#特征的选择
x = titanic[['pclass','age','sex']]
y = titanic['survived']
print(x.info())

#RangeIndex: 1313 entries, 0 to 1312
#Data columns (total 3 columns):
#pclass    1313 non-null object
#age       633 non-null float64
#sex       1313 non-null object
#dtypes: float64(1), object(2)
#可见，1)age只有633个数据，需要补全
#     2)sex和pclass变为0/1表示
  
#首先补充age里的数据，为了使影响最小，可以使用平均数或中位数

x['age'].fillna(x['age'].mean(),inplace=True)

print('处理后，数据信息为:\n')
print(x.info())

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=33)

vec = DictVectorizer(sparse = False)

#转换完成后，凡是类别型的特征都单独剥离出来，独成一列特征，数值型的则保留不变
x_train_transformed = vec.fit_transform(x_train.to_dict(orient='record'))
print(vec.feature_names_)

x_test_transformed = vec.transform(x_test.to_dict(orient='record'))

dtc = DecisionTreeClassifier()

dtc.fit(x_train_transformed,y_train)
y_predict = dtc.predict(x_test_transformed)
print('Accuracy:',dtc.score(x_test_transformed,y_test))

print(classification_report(y_predict,y_test,target_names=['died','survived']))
