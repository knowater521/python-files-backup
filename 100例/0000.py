x=[1,5,3,4,7]
y=[1,3,2,4,3]
j=0
count = 0
z=  [i for i in range(len(x))]
for i in range(4):
	if x[i]!=y[i]:
		z[j] = y[i]
		j=j+1
		count = count +1
	else:
		z[j] = 0
		j=j+1
		

	