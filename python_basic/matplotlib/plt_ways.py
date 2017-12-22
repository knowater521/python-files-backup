# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 10:55:58 2017

@author: Administrator
"""

import matplotlib.pyplot as plt
import numpy as np

n = 1024

#scatter
x = np.random.normal(0,1,n)
y = np.random.normal(0,1,n)
t = np.arctan2(y,x)  #生成颜色
plt.scatter(x,y,s=75,c=t,alpha=0.5) #alpha 透明度
plt.xlim((-1.5,1.5))
plt.ylim((-1.5,1.5))
plt.xticks(())
plt.yticks(())  #隐藏x,y脊梁

plt.show()

plt.figure()
n1 = 12
x1 = np.arange(n1)
y11 = (1-x1/float(n1)) * np.random.uniform(0.5,1.0,n1)
y12 = (1-x1/float(n1)) * np.random.uniform(0.5,1.0,n1)

plt.bar(x1,+y11,facecolor='#9999ff',edgecolor='white')
plt.bar(x1,-y12,facecolor='#ff9999',edgecolor='white')  #此处上下

for xx,yy in zip(x1,y11):        #把x1,y11分别传入xx,yy
  plt.text(xx,yy+0.05,'%.2f'%yy,ha='center',va='bottom')   #ha 横向对齐  va 纵向对齐


for xx,yy in zip(x1,y12):        
  plt.text(xx,-yy-0.05,'-%.2f'%yy,ha='center',va='top')   #ha 横向对齐  va 纵向对齐

plt.xlim(-.5,n1)
plt.xticks(())
plt.ylim(-1.25,1.25)
plt.yticks(())

plt.show()


#等高线 contours


def f(x,y):
    # the height function
    return (1 - x / 2 + x**5 + y**3) * np.exp(-x**2 -y**2)
plt.figure()
n = 256
x = np.linspace(-3, 3, n)
y = np.linspace(-3, 3, n)
X,Y = np.meshgrid(x, y)

plt.contourf(X, Y, f(X, Y), 8, alpha=.75, cmap=plt.cm.hot)  #加颜色 8 是指等高线高度分为多少份，8是10
C = plt.contour(X, Y, f(X, Y), 8, colors='black', linewidth=.5) #画线
plt.clabel(C, inline=True, fontsize=10)
plt.xticks(())
plt.yticks(())
plt.show()































