x=int(input('Please input a number no more than 99999:'))
a=x//10000
b=x%10000//1000
c=x%1000//100
d=x%100//10
e=x%10
bits = 0

if a!=0:
	bits = 5
	print('%d it is a %d bit number '%(x,bits))
	print('%d,%d,%d,%d,%d'%(e,d,c,b,a))
elif b!=0:
	bits = 4
	print('%d it is a %d bit number '%(x,bits))
	print('%d,%d,%d,%d'%(e,d,c,b))
elif c!=0:
	bits = 3
	print('%d it is a %d bit number '%(x,bits))
	print('%d,%d,%d'%(e,d,c))
elif d!=0:
	bits = 2
	print('%d it is a %d bit number '%(x,bits))
	print('%d,%d'%(e,d))
elif e!=0:
	bits = 1
	print('%d it is a %d bit number '%(x,bits))
	print('%d'%e)