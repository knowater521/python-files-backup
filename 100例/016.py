import datetime
if __name__=='__main__':
	print(datetime.date.today().strftime('%d/%m/%Y'))
	myBrithday = datetime.date(1993,6,10)
	print(myBrithday.strftime('%d/%m/%Y'))
