# class Myclass:
# 	"""一个简单的类的实例"""
# 	i = 12345
# 	def f(self):
# 		return 'hello world'


# #实例化类

# x = Myclass()

# #访问类的属性和方法

# print("Myclass 类的属性i为：",x.i)
# print("Myclass 类的方法为:",x.f())

# class Complex:
# 	def __init__(self , realpart,imagpart):
# 		self.r = realpart
# 		self.i = imagpart

# x = Complex(3.0,-4.5)
# print(x.r,x.i)
# 

# class people:
# 	#定义基本属性
# 	name = ''
# 	age = 0
# 	#定义私有属性，私有属性在类外部无法直接访问
# 	__weight = 0;
# 	#定义构造方法
# 	def __init__ (self,n,a,w):
# 		self.name = n
# 		self.age = a
# 		self.__weight = w

# 	def speak(self):
# 		print("%s说：我%d岁,%d斤。 "%(self.name,self.age,self.__weight))

# #实例化类

# p = people('吴鹏飞',24,120)
# p.speak()

########################继承
# class people:
# 	name = ''
# 	age = 0
# 	__weight = 0

# 	def __init__(self,n,a,w):
# 		self.name = n
# 		self.age = a
# 		self.__weight = w

# 	def speak(self):
# 		print("%s说：我%d岁。")

# # 单继承示例

# class student(people):
# 	"""docstring for student"""
# 	grade = ''
# 	def __init__(self, n,a,w,g):
# 		# super(student, self).__init__()
# 		# 调用父类的构造函数
# 		people.__init__(self,n,a,w)
# 		self.grade = g

# 	#覆写父类的方法
# 	def speak(self):
# 		print("%s说：我%d岁了，我在读%d年级"%(self.name,self.age,self.grade))
		

# s = student('吴鹏飞',24,120,1)
# s.speak()


# # 多继承示例
# class speaker():
# 	topic = ''
# 	name =''
# 	def __init__ (self,n,t):
# 		self.name = n
# 		self.topic = t

# 	def speak(self):
# 		# print("我叫%s，我是一个演说家，我演说的主题是%s。"%(self.name,self.topic))
# 		print("我叫%s"%self.name)


# class sample(speaker,student):   #注意，此处有顺序问题，由于speaker在左边，故调用speaker里面的speak方法
# 	a = ''
# 	def __init__ (self,n,a,w,g,t):
# 		student.__init__(self,n,a,w,g)
# 		speaker.__init__(self,n,t)

# test = sample("吴鹏飞",24,120,1,"如何用sublime text编写python程序")
# test.speak()


#类的属性和方法

# class JustCounter:
# 	__secretCount = 0  #私有变量
# 	publicCount = 0    #公有变量

# 	def count(self):
# 		self.__secretCount += 1
# 		self.publicCount += 1
# 		print(self.__secretCount)


# counter = JustCounter()
# counter.count()
# counter.count()
# print(counter.publicCount)
# # print(counter.__secretCount)  #报错，实例不能访问私有变量
# # 
# # 运算符重载

# class Vector:
# 	def __init__(self,a,b):
# 		self.a = a
# 		self.b = b


# 	def __str__(self):
# 		return "Vector(%d,%d)"%(self.a,self.b)

# 	def __add__(self,other):
# 		return Vector(self.a+other.a,self.b+other.b)


# V1 = Vector(2,10)
# V2 = Vector(5,-2)

# print(V1)
# print(V2)
# print(V1+V2)


#多线程

# import _thread
# import time

# #为线程定义一个函数
# #
# def print_time(threadName,delay):
# 	count = 0
# 	while count<5:
# 		time.sleep(delay)
# 		count += 1
# 		print("%s:%s"%(threadName,time.ctime(time.time())))

# #创建两个线程
# #

# try:
# 	_thread.start_new_thread(print_time,("Thread-1",2,))
# 	_thread.start_new_thread(print_time,("Thread-2",4,))

# except:
# 	print("Error:无法启动线程")

# while 1:
# 	pass

# import threading
# import time

# exitFlag = 0

# class myThread(threading.Thread):
# 	def __init__(self,threadId,name,counter):
# 		threading.Thread.__init__(self)
# 		self.threadId = threadId
# 		self.name = name
# 		self.counter = counter

# 	def run(self):
# 		print("开始线程："+self.name)
# 		print_time(self.name,self.counter,3)
# 		print("退出线程:"+self.name)

# def print_time(threadName,delay,counter):
# 		while counter:
# 			if exitFlag:
# 				threadName.exit()
# 			time.sleep(delay)
# 			print("%s:%s"%(threadName,time.ctime(time.time())))
# 			counter-=1
	

# thread1 = myThread(1,"Thread-1",2)
# thread2 = myThread(2,"Thread-2",2)

# thread1.start()
# thread2.start()
# thread1.join()
# thread2.join()   #把主线程阻塞在这儿
# print("退出主线程")

import threading
import time
class myThread(threading.Thread):
	"""docstring for myThread"""
	def __init__(self, threadID,name,counter):
		threading.Thread.__init__(self)
		self.threadID = threadID
		self.name = name
		self.counter = counter

	def run(self):
		print("开启线程：" + self.name)
		#获取锁，用于线程同步
		threadLock.acquire()
		print_time(self.name,self.counter,3)
		#释放锁，开启下一个进程
		threadLock.release()
		
def print_time(threadName,delay,counter):
	while counter:
		time.sleep(delay)
		print("%s:%s"%(threadName,time.ctime(time.time())))
		counter-=1

threadLock = threading.Lock()
threads = []
thread1 = myThread(1,"Thread-1",1)
thread2 = myThread(2,"Thread-2",2)

thread1.start()
thread2.start()

threads.append(thread1)
threads.append(thread2)

for t in threads:
	t.join()

print("退出主线程")

		