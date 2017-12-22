#1,2,3,5,8,13,21...

f1 = 1
f2 = 2

for i in range(10):
	print(f1,f2,end=' ' )
	f1 = f1 + f2
	f2 = f1 + f2