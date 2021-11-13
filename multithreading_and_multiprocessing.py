1, 线程与进程的区别
进程（process）和线程（thread）是操作系统的基本概念，但是它们比较抽象，不容易掌握。
关于多进程和多线程，教科书上最经典的一句话是“进程是资源分配的最小单位，线程是CPU调度的最小单位”。
线程是程序中一个单一的顺序控制流程。进程内一个相对独立的、可调度的执行单元，是系统独立调度和分派CPU的基本单位指运行中的程序的调度单位。
在单个程序中同时运行多个线程完成不同的工作，称为多线程
线程与进程的区别可以归纳为以下4点：
地址空间和其它资源（如打开文件）：进程间相互独立，同一进程的各线程间共享。某进程内的线程在其它进程不可见。
通信：进程间通信IPC，线程间可以直接读写进程数据段（如全局变量）来进行通信——需要进程同步和互斥手段的辅助，以保证数据的一致性。
调度和切换：线程上下文切换比进程上下文切换要快得多。
在多线程OS中，进程不是一个可执行的实体。

2,Python全局解释器锁GIL
全局解释器锁（英语：Global Interpreter Lock，缩写GIL），并不是Python的特性，它是在实现Python解析器（CPython）时所引入的一个概念。
由于CPython是大部分环境下默认的Python执行环境。所以在很多人的概念里CPython就是Python，也就想当然的把GIL归结为Python语言的缺陷。
那么CPython实现中的GIL又是什么呢？来看看官方的解释：
Python代码的执行由Python 虚拟机(也叫解释器主循环，CPython版本)来控制，Python 在设计之初就考虑到要在解释器的主循环中，同时只有一个线程在执行，
即在任意时刻，只有一个线程在解释器中运行。对Python 虚拟机的访问由全局解释器锁（GIL）来控制，正是这个锁能保证同一时刻只有一个线程在运行。

3,Python的多进程包multiprocessing
创建管理进程模块：
Process（用于创建进程）
Pool（用于创建管理进程池）
Queue（用于进程通信，资源共享）
Value，Array（用于进程通信，资源共享）
Pipe（用于管道通信）
Manager（用于资源共享）

同步子进程模块：
Condition（条件变量）
Event（事件）
Lock（互斥锁）
RLock（可重入的互斥锁(同一个进程可以多次获得它，同时不会造成阻塞)
Semaphore（信号量）

4, 使用示例
# -*- coding:utf-8 -*-
# Pool+map
from multiprocessing import Pool

def test(i):
    print(i)

if __name__ == "__main__":
    lists = range(100)
    pool = Pool(8)
    pool.map(test, lists)
    pool.close()
    pool.join()

# -*- coding:utf-8 -*-
# 异步进程池（非阻塞）
from multiprocessing import Pool

def test(i):
    print(i)

if __name__ == "__main__":
    pool = Pool(8)
    for i in range(100):
        '''
        For循环中执行步骤：
        （1）循环遍历，将100个子进程添加到进程池（相对父进程会阻塞）
        （2）每次执行8个子进程，等一个子进程执行完后，立马启动新的子进程。（相对父进程不阻塞）
        apply_async为异步进程池写法。异步指的是启动子进程的过程，与父进程本身的执行（print）是异步的，而For循环中往进程池添加子进程的过程，与父进程本身的执行却是同步的。
        '''
        pool.apply_async(test, args=(i,))  # 维持执行的进程总数为8，当一个进程执行完后启动一个新进程.
    print("test")
    pool.close()
    pool.join()


# -*- coding:utf-8 -*-
# 异步进程池（非阻塞）
from multiprocessing import Pool

def test(i):
    print(i)

if __name__ == "__main__":
    pool = Pool(8)
    for i in range(100):
        '''
            实际测试发现，for循环内部执行步骤：
            （1）遍历100个可迭代对象，往进程池放一个子进程
            （2）执行这个子进程，等子进程执行完毕，再往进程池放一个子进程，再执行。（同时只执行一个子进程）
            for循环执行完毕，再执行print函数。
        '''
        pool.apply(test, args=(i,))  # 维持执行的进程总数为8，当一个进程执行完后启动一个新进程.
    print("test")
    pool.close()
    pool.join()

5,Queue
get_nowait()：同q.get(False)
put_nowait()：同q.put(False)
empty()：调用此方法时q为空则返回True，该结果不可靠，比如在返回True的过程中，如果队列中又加入了项目。
full()：调用此方法时q已满则返回True，该结果不可靠，比如在返回True的过程中，如果队列中的项目被取走。
qsize()：返回队列中目前项目的正确数量，结果也不可靠，理由同q.empty()和q.full()一样

from multiprocessing import Process, Queue
import os, time, random
def write(q):
    print('Process to write: %s' % os.getpid())
    for value in ['A', 'B', 'C']:
        print('Put %s to queue...' % value)
        q.put(value)
        time.sleep(random.random())
def read(q):
    print('Process to read: %s' % os.getpid())
    while True:
        value = q.get(True)
        print('Get %s from queue.' % value)
if __name__ == "__main__":
    q = Queue()
    pw = Process(target=write, args=(q,))
    pr = Process(target=read, args=(q,))
    pw.start()
    pr.start()
    pw.join()  # 等待pw结束
    pr.terminate()  # pr进程里是死循环，无法等待其结束，只能强行终止


6.Lock（互斥锁）
Lock锁的作用是当多个进程需要访问共享资源的时候，避免访问的冲突。加锁保证了多个进程修改同一块数据时，同一时间只能有一个修改，
即串行的修改，牺牲了速度但保证了数据安全。Lock包含两种状态——锁定和非锁定，以及两个基本的方法。

from multiprocessing import Process, Lock
def l(lock, num):
    lock.acquire()
    print("Hello Num: %s" % (num))
    lock.release()
if __name__ == '__main__':
    lock = Lock()  # 这个一定要定义为全局
    for num in range(20):
        Process(target=l, args=(lock, num)).start()

RLock（可重入的互斥锁(同一个进程可以多次获得它，同时不会造成阻塞):
RLock（可重入锁）是一个可以被同一个线程请求多次的同步指令。RLock使用了“拥有的线程”和“递归等级”的概念，处于锁定状态时，RLock被某个线程拥有。
拥有RLock的线程可以再次调用acquire()，释放锁时需要调用release()相同次数。可以认为RLock包含一个锁定池和一个初始值为0的计数器，
每次成功调用 acquire()/release()，计数器将+1/-1，为0时锁处于未锁定状态。


7.Semaphore（信号量）
信号量是一个更高级的锁机制。信号量内部有一个计数器而不像锁对象内部有锁标识，而且只有当占用信号量的线程数超过信号量时线程才阻塞。
这允许了多个线程可以同时访问相同的代码区。比如厕所有3个坑，那最多只允许3个人上厕所，后面的人只能等里面有人出来了才能再进去，
如果指定信号量为3，那么来一个人获得一把锁，计数加1，当计数等于3时，后面的人均需要等待。一旦释放，就有人可以获得一把锁。

from multiprocessing import Process, Semaphore
import time, random
def go_wc(sem, user):
    sem.acquire()
    print('%s 占到一个茅坑' % user)
    time.sleep(random.randint(0, 3))
    sem.release()
    print(user, 'OK')
if __name__ == '__main__':
    sem = Semaphore(2)
    p_l = []
    for i in range(5):
        p = Process(target=go_wc, args=(sem, 'user%s' % i,))
        p.start()
        p_l.append(p)
    for i in p_l:
        i.join()


8.Python并发之concurrent.futures
Python标准库为我们提供了threading和multiprocessing模块编写相应的多线程/多进程代码。从Python3.2开始，标准库为我们提供了concurrent.futures模块，
它提供了ThreadPoolExecutor和ProcessPoolExecutor两个类，实现了对threading和multiprocessing的更高级的抽象，对编写线程池/进程池提供了直接的支持。
concurrent.futures基础模块是executor和future

from concurrent import futures
def test(num):
    import time
    return time.ctime(), num
with futures.ThreadPoolExecutor(max_workers=1) as executor:
    future = executor.submit(test, 1)
    print(future.result())


from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import pymysql

NUM = 200
PROCE_NUM = 10

def data_check(n):
    print(n)

if __name__ == '__main__':
    t = ThreadPoolExecutor(PROCE_NUM)
    threadLock = Lock()
    # 并发执行sql
    for i in range(NUM):
        obj = t.submit(data_check, i)

    t.shutdown(wait=True)