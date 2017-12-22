def IsPrime(y):
	flag = 0
	x=y//2
	while x>1:
		if y%x == 0:
			#print(y,'has factor',x)
			break
		x-=1
	else:
		#print(y,'is prime')
		flag = 1
        
	return flag
for i in range(101,201):
	if IsPrime(i):
		print(i,end=',')