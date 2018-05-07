import numpy as np


# 创建一个从0-9的一维数组
#

arr = np.arange(10)
print('arr is:',arr)


# 创建一个3*3的布尔数组
# 

bool_arr = np.full((3,3),True,dtype=bool)
print(bool_arr)


# 提取数组中的奇数元素
# 

arr1 = arr[arr%2 == 1]
print('arr中奇数元素为：',arr1)

arr2 = arr[arr%2 == 0]
print('arr中偶数元素为：',arr2)

# 替换数组中满足条件的元素
# 

# 用-1 替换奇数
# 
# 
arr[arr%2 == 1] = -1
print('替换后arr为：',arr)

# 此法会更改原数组
# 


# 以下是一种不改变原数组的方法
# 
# 
arr = np.arange(10)
out = np.where(arr%2 == 1,-1,arr)
print('arr为:',arr)
print('out为:',out)


# 垂直堆叠两个数组
# 

a = np.arange(10).reshape(2,-1)
b = np.repeat(1,10).reshape(2,-1)  # 10个1

# 法1
res1 = np.concatenate([a,b],axis=0)

# 法2
res2 = np.vstack((a,b))

# 法3
res3 = np.r_[a,b]

print('res1为：\n',res1)
print('res2为：\n',res2)
print('res3为：\n',res3)


# 对应的水平堆叠分别为
# np.concatenate([a,b],axis=1)
# np.hstack((a,b))
# np.c_[a,b]
# 


print(np.tile(a,(3,2)))  # 水平方向复制2次，垂直方向复制3次 在numpy中axis=0是垂直方向

#  获取两个数组中相同的元素
#  


a = np.array([1,2,3,2,3,4,3,4,5,6])
b = np.array([7,2,10,2,7,4,9,4,9,8])

res = np.intersect1d(a,b)
print('a与b中相同的元素为：',res)


# 从一个数组a中删除在数组b中存在的所以元素，即非的关系
# 

a = np.array([1,2,3,4,5])
b = np.array([5,6,7,8,9])

res = np.setdiff1d(a,b)
print(res)


# 获取数组a和b的元素匹配的索引号
# 

a = np.array([1,2,3,2,3,4,3,4,5,6])
b = np.array([7,2,10,2,7,4,9,4,9,8])
print(np.where(a==b))


# 从数组中提取给定范围内的数字
# 

a = np.arange(15)
index = np.where( (a>=5) & (a<=10) )  # index = np.where(np.logical_and(a>=5,a<=10))
print('a中数据满足大于等于5，小于等于10的有：',a[index])









































