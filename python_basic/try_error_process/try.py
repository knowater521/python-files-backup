
# try 用来处理错误，比如读取某个不存在的文件，会报错
# 

# file = open('eeee','r')   #会报错

try:
	file1 = open('eeeeee','r+')
except Exception as e:      #如果打开失败，把错误信息存在e中
	print(e)
	print('There is no file named as eeeeee')
	reponse = input('do you want to create a new file? y/n')
	if reponse == 'y':
		file1 = open('eeeeee','w')
	else:
		pass
else:                #如果打开成功，直接跳到此处
	file1.write('hello world')
file1.close()
