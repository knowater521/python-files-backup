# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 22:23:04 2017

@author: Administrator
"""

import pandas as pd
import numpy as np

data = pd.read_csv('01.csv')
print(data)
#怎么改index呢？？

data.to_pickle('01.pickle')

data1 = pd.read_pickle('01.pickle')

####合并不同的dataframe
##第一种concat
#df1 = pd.DataFrame(np.ones((3,4))*0, columns=['a','b','c','d'])
#df2 = pd.DataFrame(np.ones((3,4))*1, columns=['a','b','c','d'])
#df3 = pd.DataFrame(np.ones((3,4))*2, columns=['a','b','c','d'])
#
#print(df1)
#print(df2)
#print(df3)
#
#res = pd.concat([df1,df2,df3],axis = 0)  #这样不会改变index
#res1 = pd.concat([df1,df2,df3],axis = 0,ignore_index=True) #改变index
#
#print(res)
#print(res1)
##  join ['inner','outer']
##注意行和列都不完全相同
#df4 = pd.DataFrame(np.ones((3,4))*0, columns=['a','b','c','d'], index=[1,2,3])
#df5 = pd.DataFrame(np.ones((3,4))*1, columns=['b','c','d','e'], index=[2,3,4])
#
#res2 = pd.concat([df4,df5])  #这种方法默认是'outer'，不同的部分为NaN
#res3 = pd.concat([df4,df5],join='inner',ignore_index=True) #只合并公有的部分
#print(res2)
#print(res3)
#
## join_axes 
#
##append
#res4 = df4.append([df5,df5],ignore_index=True)
#print(res4)
#
#s1 = pd.Series([1,2,3,4],index=['a','b','c','d'])
#res5 = df4.append(s1,ignore_index=True)   #添加一行
#print(res5)

##第二种merge

#基于某一个key
#left = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'],
#                             'A': ['A0', 'A1', 'A2', 'A3'],
#                             'B': ['B0', 'B1', 'B2', 'B3']})
#right = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'],
#                              'C': ['C0', 'C1', 'C2', 'C3'],
#                              'D': ['D0', 'D1', 'D2', 'D3']})
#
#print(left)
#print(right)
#
#res = pd.merge(left,right,on='key') #基于'key'合并
#print(res)
#
##考虑两组Key
#
##注意两组key不完全相同
#left = pd.DataFrame({'key1': ['K0', 'K0', 'K1', 'K2'],
#                      'key2': ['K0', 'K1', 'K0', 'K1'],
#                      'A': ['A0', 'A1', 'A2', 'A3'],
#                      'B': ['B0', 'B1', 'B2', 'B3']})
#right = pd.DataFrame({'key1': ['K0', 'K1', 'K1', 'K2'],
#                       'key2': ['K0', 'K0', 'K0', 'K0'],
#                       'C': ['C0', 'C1', 'C2', 'C3'],
#                       'D': ['D0', 'D1', 'D2', 'D3']})
#print(left)
#print(right)
## how =['left','right','outer','inner']
#res = pd.merge(left,right,on=['key1','key2'])  #默认是inner方法，即key1,key2完全相同的保留
#print(res)
#res = pd.merge(left,right,on=['key1','key2'],how='outer') 
#print(res)

## 基于index合并

# 1
#    age_boy   k  age_girl
# 0        1  K0         4






























