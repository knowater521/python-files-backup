# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 14:44:39 2017

@author: Administrator
"""

from sklearn import tree
from sklearn.datasets import load_iris
#import graphviz
X = [[0,0],[1,1]]
Y=[0,1]
clf = tree.DecisionTreeClassifier();
clf = clf.fit(X,Y)

#After fitted, the model can then be used to predict the class of samples
print(clf.predict([[2,2]]))  # [1]

print(clf.predict_proba([[2,2]])) #the probability of each label


iris = load_iris()
clf = tree.DecisionTreeClassifier()
clf = clf.fit(iris.data,iris.target)

#dot_data = tree.export_graphviz(clf,out_file=None)
#graph = graphviz.Source(dot_data)
#graph.render("iris")

