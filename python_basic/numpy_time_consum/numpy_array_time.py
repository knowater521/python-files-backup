import numpy as np
import time 


a = np.zeros((200,200),order='C') #以C语言形式存储（以行为主）
b = np.zeros((200,200),order='F') #以Fortan形式存储（以列为主）
N = 9999

def f1(a):
	for _ in range(N):
		np.concatenate((a,a),axis = 0)  #以行合并

def f2(a):
	for _ in range(N):
		np.concatenate((a,a),axis = 0)

def f3(a):
	for _ in range(N):
		np.vstack((a,a,))  #以vstack的形式合并行

t0 = time.time()
f1(a)
t1 = time.time()
f2(b)
t2 = time.time()
f3(a)
t3 = time.time()

print('C形式用时为：',(t1-t0))
print('Fortan形式用时为：',(t2-t1))

print('concatenate形式用时为：',t1-t0)
print('vstack形式用时为：',t3-t2)