height = 100.00

def Fun(height,n):
	sum = 0
	for i in range(n):
		if i==0:
			sum+=height
		else:
			sum+=height*2
		height/=2
	
	print('sum=%lf,height=%lf'%(sum,height))
		
Fun(height,2)