def Factorial(n):
	if n == 1 or n==0:
		fn=1
	else:
		fn = n*Factorial(n-1)
	return fn
	
print(Factorial(5))