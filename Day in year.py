def Isleap(year):
	leap = 0;
	if (year % 400 == 0) or ((year % 4 == 0) and (year % 100 != 0)):
		leap = 1
		
	return leap
	
months = (0,31,59,90,120,151,181,212,243,273,304,334)

Date_Str=input('Please input the date in format of YYYYMMDD:')
[year,month,day] = [int(Date_Str[:4]),int(Date_Str[4:6]),int(Date_Str[6:8])]
sum = 0
leap = Isleap(year)
if 0 < month <=12 and 0<day <=31:
	sum = months[month-1]+day
else:
	print('Please check your date')
	
	
leap = Isleap(year)
if leap and month>2 :
	sum+=1
	
print(Date_Str,'is the %dth day.'%(sum))