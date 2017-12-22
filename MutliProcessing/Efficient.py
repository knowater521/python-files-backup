import multiprocessing as mp
import threading as td
import time

def job(q):
	res = 0
	for i in range(100000):
		res += i + i**2 + i**3
	q.put(res)


def multicore():
	q = mp.Queue()
	p1 = mp.Process(target=job,args=(q,))
	p2 = mp.Process(target=job,args=(q,))
	p1.start()
	p2.start()
	p1.join()
	p2.join()
	res1 = q.get()
	res2 = q.get()
	print("multicore:",res1+res2)

def normal():
	res = 0
	for _ in range(2):
		for i in range(100000):
			res += i + i**2 + i**3
	print('normal:',res)

def multithread():
	q = mp.Queue()
	t1 = td.Thread(target=job,args=(q,))
	t2 = td.Thread(target=job,args=(q,))
	t1.start()
	t2.start()
	t1.join()
	t2.join()
	res1 = q.get()
	res2 = q.get()
	print('thread:',res1+res2)

if __name__=='__main__':
	starttime = time.time()
	normal()
	endtime1 = time.time()
	print("normal time:",endtime1 - starttime)

	multithread()
	endtime2 = time.time()
	print("multithread time:",endtime2 - endtime1)

	multicore()
	endtime3 = time.time()
	print("multicore time:",endtime3 - endtime2)

