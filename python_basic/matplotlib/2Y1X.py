# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 19:56:30 2017

@author: Administrator
"""


#主次坐标轴
import matplotlib.pyplot as plt
import numpy as np

x = np.arange(0,10,0.1)
y1 = 0.05 * x ** 2
y2 = -y1


fig,ax1 = plt.subplots()
ax2 = ax1.twinx()   #ax2 是 ax1的镜面反射
ax1.plot(x,y1,'g-')
ax2.plot(x,y2,'b-')

ax1.set_xlabel('x data')
ax1.set_ylabel('y1',color='g')
ax2.set_ylabel('y2',color='b')
plt.show()
