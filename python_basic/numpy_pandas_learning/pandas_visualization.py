# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 23:09:31 2017

@author: Administrator
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Series
data = pd.Series(np.random.randn(1000),index=np.arange(1000))
data = data.cumsum() #累加数据
data.plot()
plt.show()

data = pd.DataFrame(np.random.randn(1000,4),index=np.arange(1000),
                    columns=list('ABCD'))
data = data.cumsum() #累加数据
print(data.head())
data.plot()
plt.show()

ax = data.plot.scatter(x='A',y='B',color='DarkBlue',label='Class1')
data.plot.scatter(x='A',y='C',color='LightGreen',label='Class2',ax=ax)
plt.show()