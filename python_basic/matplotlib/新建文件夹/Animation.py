# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 20:01:16 2017

@author: Administrator
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
#import matplotlib
#matplotlib.rcParams['animation.convert_path'] = 'D:\\software\\ImageMagick-7.0.7-Q16\\convert.exe'
fig,ax = plt.subplots()

x = np.arange(0,2*np.pi,0.01)
line, = ax.plot(x,np.sin(np.pi * x))

ax.set_ylim((-2,2))
#ax = plt.axes(xlim=(0,7),ylim=(-2,2))
#label = ax.text([],[],'')

y = np.sin(np.pi * x)
yy = []

for i in range(100):
  k = 2*i+1
  fk = 2 / ( k * np.pi)
  yy.append(fk * np.sin(k * np.pi * x))

def f1(i):
  line.set_ydata(np.sin(np.pi *  x+i/10))
  return line,


def init_fun():
  
  line.set_ydata(y)
  return line,


def update(i):
#  global label,ax
  
  plt.text(3,1.5,r'$K=%d$'%(i+1),
         fontdict={'size': 16, 'color': 'r'})
#  ax.text(3,1.5,'K=%d'%(i+1))
#  label.set_text('k=%d'%(i+1))
#  label.set_position(['3','1.5'])
  line.set_ydata(sum(yy[0:i]) )
#  ax.text(3,1.5,' ')
  return line,
  
ani = animation.FuncAnimation(fig=fig,func = update ,frames=10,  #其中一种，其他的自行搜索
                              init_func=init_fun,interval=200,repeat=True,blit=True)  


ani.save('ani.gif',writer='imagemagick',fps=2)

plt.show()
#import numpy as np
#import matplotlib.pyplot as plt
#from matplotlib.animation import FuncAnimation
#
## Fixing random state for reproducibility
#np.random.seed(19680801)
#
#
## Create new Figure and an Axes which fills it.
#fig = plt.figure(figsize=(7, 7))
#ax = fig.add_axes([0, 0, 1, 1], frameon=False)
#ax.set_xlim(0, 1), ax.set_xticks([])
#ax.set_ylim(0, 1), ax.set_yticks([])
#
## Create rain data
#n_drops = 50
#rain_drops = np.zeros(n_drops, dtype=[('position', float, 2),
#                                      ('size',     float, 1),
#                                      ('growth',   float, 1),
#                                      ('color',    float, 4)])
#
## Initialize the raindrops in random positions and with
## random growth rates.
#rain_drops['position'] = np.random.uniform(0, 1, (n_drops, 2))
#rain_drops['growth'] = np.random.uniform(50, 200, n_drops)
#
## Construct the scatter which we will update during animation
## as the raindrops develop.
#scat = ax.scatter(rain_drops['position'][:, 0], rain_drops['position'][:, 1],
#                  s=rain_drops['size'], lw=0.5, edgecolors=rain_drops['color'],
#                  facecolors='none')
#
#
#def update(frame_number):
#    # Get an index which we can use to re-spawn the oldest raindrop.
#    current_index = frame_number % n_drops
#
#    # Make all colors more transparent as time progresses.
#    rain_drops['color'][:, 3] -= 1.0/len(rain_drops)
#    rain_drops['color'][:, 3] = np.clip(rain_drops['color'][:, 3], 0, 1)
#
#    # Make all circles bigger.
#    rain_drops['size'] += rain_drops['growth']
#
#    # Pick a new position for oldest rain drop, resetting its size,
#    # color and growth factor.
#    rain_drops['position'][current_index] = np.random.uniform(0, 1, 2)
#    rain_drops['size'][current_index] = 5
#    rain_drops['color'][current_index] = (0, 0, 0, 1)
#    rain_drops['growth'][current_index] = np.random.uniform(50, 200)
#
#    # Update the scatter collection, with the new colors, sizes and positions.
#    scat.set_edgecolors(rain_drops['color'])
#    scat.set_sizes(rain_drops['size'])
#    scat.set_offsets(rain_drops['position'])
#
#
## Construct the animation, using the update function as the animation
## director.
#animation = FuncAnimation(fig, update, interval=10)
#plt.show()
