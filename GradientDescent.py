# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 18:42:45 2017

@author: Administrator
"""
import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt
def Sgn(x):
  if x>0:
    return 1
  else:
    return 0

#xx=np.linspace(0,4.5,100)
#yy=xx
x=np.array([[1,2,3,2.5,4.5,4,3,1],
           [2,1,2,2,3.2,6,8,7]]
           )
y=[1,-1,-1,-1,-1,1,1,1]
#y=[i for i in range(len(x[0]))]
#plt.scatter(x[0],x[1])
#plt.plot(xx,yy)
w=[0,0]
b=0
count = 0  #y[n] 和 fx[n]不相等的个数
index_ineq = 0
matrix_ineq=
fx=[0 for n in range(len(x[0]))]

for i in range(len(x[0])):
  fx[i]=Sgn(np.dot(w,x[:,i])+b)

for index_ineq in range(len(x[0])):
  if fx[index_ineq]!=y[index_ineq]:
    count = count + 1
    
dR_dw=np.zeros((2,count),dtype=np.float)
for index_ineq in range(len(x[0])):
  if fx[index_ineq]!=y[index_ineq]:
    
    

