# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import imageio
import numpy as np
import os


def SaveMkdir(dir):
	if not os.path.exists(dir):
		os.mkdir(dir)



def MakeImage():
	# 自变量
	x = np.arange(0,2*np.pi,0.01)

	# 各阶次谐波
	
	yy = []
	for i in range(100):
	     k = 2*i+1
	     fk = 2 / ( k * np.pi)
	     yy.append(fk * np.sin(k * np.pi * x))

	for i in range(1,100):
		line = np.sum(yy[0:i],axis=0)
		plt.cla()
		plt.plot(x,line)
		plt.text(3.2,0.8,'Step%d'%i,fontdict={'size':10,'color':'red'})
		plt.ylim([-1,1])
		plt.savefig('img/{}.png'.format(i))



def SaveGif():
	# save as gif use imageio
	images = []
	for i in range(1,100):
		img_name = 'img/{}.png'.format(i)
		images.append(imageio.imread(img_name))


	imageio.mimsave('square_wave_synthesis.gif',images,fps=5)


if __name__ == '__main__':
	SaveMkdir('./img')
	MakeImage()
	SaveGif()








