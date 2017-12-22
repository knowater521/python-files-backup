# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 19:48:15 2017

@author: Administrator
"""

import matplotlib.pyplot as plt 


#图中图

fig = plt.figure()
x = list(range(1,8))
y = [1,3,4,2,5,8,6]

left,bottom,width,height=0.1,0.1,0.8,0.8
ax1 = fig.add_axes([left,bottom,width,height])
ax1.plot(x,y,'r')
ax1.set_xlabel('x')
ax1.set_xlabel('y')
ax1.set_title('title')

left,bottom,width,height=0.2,0.6,0.25,0.25    #定位
ax2 = fig.add_axes([left,bottom,width,height])
ax2.plot(x,y,'b')
ax2.set_xlabel('x')
ax2.set_xlabel('y')
ax2.set_title('inside1')


plt.axes([0.6,0.2,0.25,0.25])
plt.plot(y[::-1],x,'g')  #y逆序
plt.xlabel('x')
plt.ylabel('y')
plt.title('inside2')

plt.show()
