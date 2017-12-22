import threading
import time
def thread_job():
	# print("This is an added Thread,number is %s"%threading.current_thread())
	print('T1 start')
	for i in range(50):
		time.sleep(0.1)

	print('T1 finished')


def T2_job():
	print('T2 start')
	print('T2 finished')



def main():
	added_thread = threading.Thread(target = thread_job,name = 'T1') 
	thread2 = threading.Thread(target = T2_job,name = 'T2')
	added_thread.start() #start the thread
	thread2.start()
	added_thread.join()  #join in mainthread
	print('all done')
	# print(threading.active_count())  #print the number of actived thread
	# print(threading.enumerate())     #list the threads
	# print(threading.current_thread()) 


if __name__ == '__main__':
	main()