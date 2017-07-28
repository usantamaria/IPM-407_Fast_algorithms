# Class to manipulate 2D source/target points
from math import sqrt


class point2d():
	def __init__(self, x, y, gamma=0, idx=-1):
		"""
		x: x axis
		y: y axis
		gamma: field
		idx: index
		"""
		self.x = x
		self.y = y
		self.gamma = gamma
		self.idx = idx
		
	def __str__(self):
		return '(%.2f,%.2f,%.2f,%d)' %(self.x,self.y,self.gamma,self.idx)
	
	# Arithmetic manipulation, only deal with spatial (x,y)
	def __add__(self,other):
		new_x = self.x + other.x
		new_y = self.y + other.y
		
		return point2d(new_x,new_y,self.gamma,self.idx)
	
	def __sub__(self,other):
		new_x = self.x - other.x
		new_y = self.y - other.y
		
		return point2d(new_x,new_y,self.gamma,self.idx)
	
	# Multiplication by scalar
	def __mul__(self, a):
		new_x = self.x*a
		new_y = self.y*a
		
		return point2d(new_x,new_y)
	
	# Division by scalar
	def __truediv__(self,b):
		new_x = self.x / b
		new_y = self.y / b
		
		return point2d(new_x,new_y)
	
	# 2D point norm
	def norm(self):
		return sqrt(self.x**2 + self.y**2)

	# assume a is a multi-index (a1,a2)
	def __pow__(self,a):
		new_x = self.x**a[0]
		new_y = self.y**a[1]
		
		return new_x*new_y