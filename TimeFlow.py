import time 
while 1:
	print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
	time.sleep(1)
