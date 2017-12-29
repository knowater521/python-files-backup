

a = [1,2,3]
b = [4,5,6]

print(zip(a,b))   #是一个object,无法打印具体内容

print(list(zip(a,b))) #转为List打印，zip做的是把a,b的相同位合并

for i,j in zip(a,b):
	print(i/2,j*2)

#zip可以合并多个，比如zip(a,a,b)
#

def fun1(x,y):
	return (x+y)

print(fun1(2,3))

#lambda代替
#

fun2 = lambda x,y:x+y
print(fun2(6,3))

#map把已知功能和参数绑定一起运算
#

print(list(map(fun1,[1],[10])))

print(list(map(fun1,[1,3,4],[10,3,4])))