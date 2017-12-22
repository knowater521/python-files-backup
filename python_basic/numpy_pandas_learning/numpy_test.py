import numpy as np
# array = np.array([[1,2,3],[3,4,5]])

# print(array)
# print('number of dim:',array.ndim)
# print('shape:',array.shape)
# print('size:',array.size)

# print("sum of array ",np.sum(array))
# print("sum of axis=0 ",np.sum(array,axis=0))
# print("sum of axis=1 ",np.sum(array,axis=1))
# print(np.argmin(array))
# print(np.argmax(array,axis = 0))


#######################
# A = np.array([1,1,1])[:,np.newaxis]
# B = np.array([2,2,2])[:,np.newaxis]
# print(A)
# print(B)
# C = np.vstack((A,B))
# D = np.hstack((A,B))
# print(C)
# print(D)

# E = np.concatenate((A,B,A,B),axis =0)
# print(E)
# 

A = np.arange(12).reshape(3,4)
print(A.flatten())   #####see it 
print(A)
print(np.split(A,2,axis=1))    # equal split
print(np.array_split(A,2,axis=0)) #inequal split
print(np.vsplit(A,3))
print(np.hsplit(A,2))

