# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 11:39:31 2017

@author: Administrator
"""

import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec
#普通方法
plt.figure()

plt.subplot(2,2,1)  #subplot(221)也可以
plt.plot([0,1],[0,1])

plt.subplot(2,2,2)
plt.plot([0,1],[0,2])

plt.subplot(2,2,3) 
plt.plot([0,3],[0,3])

plt.subplot(224)
plt.plot([0,1],[0,4])

plt.figure()

plt.subplot(211)   #两行一列
plt.plot([0,1],[0,1])

plt.subplot(234)  #注意数字，两行3列
plt.plot([0,1],[0,2])

plt.subplot(235) 
plt.plot([0,3],[0,3])

plt.subplot(236)
plt.plot([0,1],[0,4])
plt.show()

#method 1 :subplot2grid

plt.figure()
ax1 = plt.subplot2grid((3,3),(0,0),colspan=3,rowspan=1) #3行3列,从第0行第0列开始画,占1行3列
ax1.plot([1,2],[1,2])
ax1.set_title('ax1_title')
ax2 = plt.subplot2grid((3,3),(1,0),colspan=1,rowspan=2)
ax3 = plt.subplot2grid((3,3),(1,1),colspan=2,rowspan=1)
ax4 = plt.subplot2grid((3,3),(2,1),colspan=1,rowspan=1)
ax5 = plt.subplot2grid((3,3),(2,2),colspan=1,rowspan=1)


# method 2 ：import gridspec
plt.figure()
gs = gridspec.GridSpec(3,3)  #3行3列
ax6 = plt.subplot(gs[0,:])   #第0行的所有列
ax7 = plt.subplot(gs[1,:2])  #第1行前两列
ax8 = plt.subplot(gs[1:,2])  #最后一列，2，3行
ax9 = plt.subplot(gs[-1,0])  #最后一行，第一列
ax9.set_title('ax_9')
ax10 = plt.subplot(gs[-1,-2]) #最后一行，第二列


#method3 :plt.subplots()

f,((ax11,ax12,ax13),(ax21,ax22,ax23)) = plt.subplots(2,3,sharex=True,sharey=True)
ax11.scatter([1,2],[1,2])
plt.show()






















