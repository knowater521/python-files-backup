def Isleap(year):
	leap = 0;
	if (year % 400 == 0) or ((year % 4 == 0) and (year % 100 != 0)):
		leap = 1
		
	return leap
c=0
while c != '#' :
	c=input('Please input the year:')
	if '0'<=c<='9':
		year = int(c)
	else:
		break
	leap = Isleap(year)
	if leap:
		print(year,'is leap year')
	else :
		print(year,'is not leap year')