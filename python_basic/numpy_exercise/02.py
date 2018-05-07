import numpy as np

# 将处理标量的函数在数组上运行
# 

def max(x,y):
	return x if x>=y else y

# 该函数是处理标量的函数，现假设输入是两个数组，分别返回对应位置的最大值
# 

a = np.array([5,7,9,8,6,4,5])
b = np.array([6,3,4,8,9,7,1])
pair_max = np.vectorize(max)
print(pair_max(a,b))


# 17 交换二维数组中的两个行
# 

arr = np.arange(9).reshape(3,3)
print('交换前,arr是:\n',arr)
arr_ = arr[[1,0,2],:]
print('交换后,arr_是:\n',arr_)


# 反转二维数组的行
# 

arr_row_transfer = arr[::-1]
print('反转行后,arr变为:\n',arr_row_transfer)


# 反转列
# 

arr_column_transfer = arr[:,::-1]
print('反转列后,arr变为:\n',arr_column_transfer)



# 创建一个5-10之间的随机数组，大小为5*3
# 


# 法1
# 
rand_arr = np.random.randint(low=5,high=10,size=(5,3))  + np.random.random((5,3))   # 若不加后面的，是整数
print(rand_arr)


# 法2
# 

rand_arr = np.random.uniform(5,10,size=(5,3))
print(rand_arr)



#  np.genfromtxt
#  

user_dat = np.genfromtxt('users.dat0',dtype="i8,S5,i8,i8,i8",delimiter='::')
# user_dat = np.loadtxt('users.dat0',dtype="i8,S5,i8,i8,i8",delimiter='::')  # 对于字符无法加载
print(user_dat)





























