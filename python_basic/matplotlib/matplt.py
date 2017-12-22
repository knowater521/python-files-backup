# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 22:27:43 2017

@author: Administrator
"""

import matplotlib.pyplot as plt
import numpy as np
#
#x = np.linspace(-1,1,50)
#y = 2 * x +1
#y1 = x ** 2
#
##修改坐标轴的描述等
#plt.plot(x,y)
#plt.plot(x,y1,color='red',linestyle='--')
#plt.xlim((-2,1.5))
#
#new_ticks = np.linspace(-1,2,5)
#plt.xticks(new_ticks)
#plt.yticks([-2,-1.8,-1,1.22,3],[r'$really\ bad$',r'$bad$',r'$normal$',r'$good$',
#           r'$\alpha$'])  
#
##修改坐标轴位置等  gca = get current axis

x = np.linspace(-3, 3, 50)
y = 2*x + 1

plt.figure(num=1, figsize=(8, 5),)
plt.plot(x, y,)

#移动坐标轴

ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')   #x轴用底部的脊梁代替
ax.spines['bottom'].set_position(('data', 0)) #x轴的位置是y为的0的位置
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data', 0))

#标注
x0 = 1
y0 = 2*x0 + 1
plt.plot([x0, x0,], [y0, 0,], 'k--', linewidth=2.5)
# set dot styles
plt.scatter([x0, ], [y0, ], s=50, color='b')   #只显示一个点
plt.annotate(r'$2x+1=%s$' % y0, xy=(x0, y0), xycoords='data', xytext=(+30, -30),
             textcoords='offset points', fontsize=16,
             arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=.2"))

plt.text(-3.7, 3, r'$This\ is\ the\ some\ text. \mu\ \sigma_i\ \alpha_t$',
         fontdict={'size': 16, 'color': 'r'})




