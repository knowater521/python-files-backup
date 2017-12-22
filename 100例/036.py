import math
def IsPrime(x):
	if x == 2:
		flag = 1
	elif x==1:
		flag = 0
	else:
		# y = int(math.sqrt(x))
		y=x//2
		for i in range(2,y+1):
			if x%i == 0:
				flag = 0
				break
		else:
			flag = 1
	return flag
		
for i in range(1,100):
	if IsPrime(i):
		print(i)