import numpy as np
url= 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

# 在数组中随机插入值(20个)
iris_1d = np.genfromtxt(url,delimiter=',',dtype=object)
i,j = np.where(iris_1d)   # i,j分别是行和列的索引 i:0-149 j:0-4 则i,j都有150*5个元素

# print(iris_1d)
# Method 1
# 
np.random.seed(100)
iris_1d[np.random.choice((i),20),np.random.choice((j),20)] = np.nan
print(iris_1d)
# 



# 查找缺失的值
# 

# print("缺失的值数量为：",np.isnan(iris_1d[:,0]).sum())
# print("缺失值的位置为：",np.where(np.isnan(iris_1d[:,0])))





a = '{0},love you {1}'.format('mama','yes')
print(a)
b = '{first} or {second},that is {third}'.format(first='to be',second='not to be ',third='a question')
print(b)









