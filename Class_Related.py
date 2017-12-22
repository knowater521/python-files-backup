# import numpy as np

class FirstClass():
	def setdata(self,value):
		self.data=value
	def display(self):
		print(self.data)
		
# x=FirstClass()
# y=FirstClass()

# x.data = 3
# x.display()
# x.setdata(4)
# x.display()

class SecondClass(FirstClass):  #SecondClass is a subclass of FirstClass,
	def display(self):          #FirstClass is a superclass of SecondClass and so on
		print('Current value = "%s"'%self.data)

# z=SecondClass()
# z.data=3
# z.display()

class ThirdClass(SecondClass):
	def __init__(self,value):
		self.data = value
	def __add__(self,other):
		return ThirdClass(self.data + other )
	def __str__(self):
		return '[ThirdClass:%s]'%self.data
	def mul(self,other):
		self.data *= other
		
a=ThirdClass('abc')
a.display()
b=a+'xyz'
b.display()
print(b)
a.mul(3)
print(a)



