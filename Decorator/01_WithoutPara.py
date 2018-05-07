import logging




def dec(func):

    def wrapper(*args,**kwargs):    # *args 表示可变参数  **kwargs 表示关键字参数
        logging.basicConfig(level=logging.DEBUG, format='%(pathname)s - %(lineno)s - %(asctime)s - %(name)s - %(levelname)s - %(message)s',
                            filename='log.txt',filemode='a+')
        logging.warning("%s is running"%func.__name__)
        logging.debug("This is debug information")
        return func(*args,**kwargs)  # 此处如果不加括号，不执行func函数

    return wrapper


@dec   # 加此语句可避免 bar = dec(bar)等的赋值

def bar():
    print("I'm bar")

@dec 
def foo():
    print("I'm foo")

if __name__ == '__main__':
    bar()
    foo()