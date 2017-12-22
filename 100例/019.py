def PerfectNum(num):
	n=-1
	k=[]
	s = num
	for i in range(1,num):
		if num%i == 0:
			n+=1
			s-=i
			k.append(i)
	if s == 0:
		print('%d='%num,end=' ')
		for x in range(n+1):
			if x == n:
				print('%d'%k[x])
			else:
				print('%d+'%k[x],end=' ' )
		
	
		
for i in range(2,1001):
	PerfectNum(i)
	
