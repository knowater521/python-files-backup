# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 11:21:57 2017

@author: Administrator
"""

import matplotlib.pyplot as plt
import numpy as np

a = np.array([0.313660827978, 0.365348418405, 0.423733120134,
              0.365348418405, 0.439599930621, 0.525083754405,
              0.423733120134, 0.525083754405, 0.651536351379]).reshape(3,3)

plt.imshow(a, interpolation='nearest', cmap='bone', origin='lower')
plt.colorbar(shrink=.92)  #压缩图例

plt.xticks(())
plt.yticks(())
plt.show()