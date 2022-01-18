import datetime


def cal_time(func):
    def wrapper(*args, **kwargs):
        t1 = time.time()
        result = func(*args, **kwargs)
        t2 = time.time()
        print("%s running time: %s secs." % (func.__name__, t2 - t1))
        return result

    return wrapper


# 1.顺序查找
def sercher_sequence(alist, item):
    find = False
    cur = 0
    while cur < len(alist):
        if alist[cur] == item:
            find = True
            break
        else:
            cur += 1
    return find


# 1,二分法查找
"""二分查找则是从中间元素开始，而不是按顺序查找列表，如果该元素是我们正在寻找的元素，我们就完成了查找。 
如果它不是，我们可以使用列表的有序性质来消除剩余元素的一半"""


def sercher_dichotomy(alist, item):
    left = 0
    right = len(alist) - 1
    find = False

    while left <= right:
        mid_index = (left + right) // 2
        if alist[mid_index] == item:
            find = True
            break
        elif alist[mid_index] > item:
            right = mid_index - 1
        else:
            left = mid_index + 1
    return find

# 递归实现
def binarysearch(alist, item):
    if len(alist) == 0:
        return False
    else:
        mid = len(alist) // 2
        if alist[mid] == item:
            return True
        else:
            if alist[mid] > item:
                return binarysearch(alist[:mid], item)
            else:
                return binarysearch(alist[mid + 1:], item)

# 3.冒泡排序

"""比较相邻的元素。如果第一个比第二个大，就交换他们两个。
对每一对相邻元素做同样的工作，从开始第一对到结尾的最后一对。在这一点，最后的元素应该会是最大的数。
针对所有的元素重复以上的步骤，除了最后一个。
持续每次对越来越少的元素重复上面的步骤，直到没有任何一对数字需要比较。"""


def bubble_sort(alist):
    for i in range(0, len(alist) - 1):
        for j in range(0, len(alist) - i - 1):
            print(i)
            print(j)
            if alist[j] > alist[j + 1]:
                alist[j], alist[j + 1] = alist[j + 1], alist[j]
    return alist


# 4.选择排序
"""第一次从待排序的数据元素中选出最小（或最大）的一个元素，存放在序列的起始位置，然后再从剩余的未排序元素中寻找到最小（大）元素，
然后放到已排序的序列的末尾。以此类推，直到全部待排序的数据元素的个数为零。选择排序是不稳定的排序方法"""


def select_sort(alist):
    for i in range(0, len(alist)):
        min_index = i
        for j in range(i + 1, len(alist)):
            if alist[j] < alist[i]:
                min_index = j
        alist[min_index], alist[i] = alist[i], alist[min_index]
    return alist


# 5.插入排序
"""基本思想是，每步将一个待排序的记录，按其关键码值的大小插入前面已经排序的文件中适当位置上，直到全部插入完为止。关键码是数据元素中某个数据项的值，
用它可以标示一个数据元素"""


def insert_sort(alist):
    for i in range(1, len(alist)):
        tmp = alist[i]
        j = i - 1
        while j >= 0 and tmp < alist[j]:
            alist[j + 1] = alist[j]
            j -= 1
        alist[j + 1] = tmp
    return alist


# 6.快速排序
"""# 基本思想是：通过一趟排序将要排序的数据分割成独立的两部分，其中一部分的所有数据都比另外一部分的所有数据都要小，
然后再按此方法对这两部分数据分别进行快速排序，整个排序过程可以递归进行，以此达到整个数据变成有序序列
优越点：分治思想的排序在处理大数据集量时效果比较好，小数据集性能差些
"""


# 1
def quick_sort(alist):
    if len(alist) <= 1:
        return alist
    mid_value = alist[(len(alist) // 2)]
    left, right = [], []
    # 防止死循环
    alist.remove(mid_value)
    for i in alist:
        if mid_value > i:
            left.append(i)
        else:
            right.append(i)
    return quick_sort(left) + [mid_value] + quick_sort(right)


# 2
def quick_sort1(alist):
    quick_sort2(alist, 0, len(alist) - 1)
    return alist


def quick_sort2(alist, lift, right):
    if right > lift:
        mid = quick_sort3(alist, lift, right)
        quick_sort2(alist, lift, mid - 1)
        quick_sort2(alist, mid + 1, right)


def quick_sort3(alist, lift, right):
    tmp = alist[lift]
    while True:
        while right > lift and alist[right] >= tmp:
            right = right - 1
        alist[lift] = alist[right]
        while right > lift and alist[lift] <= tmp:
            lift = lift + 1
        alist[right] = alist[lift]
        if right == lift:
            break
    alist[lift] = tmp
    return lift


# 7.归并排序
"""归并排序主要通过先递归地将数组分为两个部分，排序后，再将元素合并到一起。
所以归根结底归并排序就是两部分组成：拆分+合并！"""


#
def merge_sort(array):
    if len(array) < 2:
        return array
    mid = len(array) // 2
    left = merge_sort(array[:mid])
    right = merge_sort(array[mid:])
    i, j = 0, 0
    res = []
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            res.append(left[i])
            i += 1
        else:
            res.append(right[j])
            j += 1
    return res + left[i:] + right[j:]


# 2
def merge(li, low, mid, high):
    i = low
    j = mid + 1
    ltmp = []
    while i <= mid and j <= high:
        if li[i] < li[j]:
            ltmp.append(li[i])
            i += 1
        else:
            ltmp.append(li[j])
            j += 1
    while i <= mid:
        ltmp.append(li[i])
        i += 1
    while j <= high:
        ltmp.append(li[j])
        j += 1
    li[low:high + 1] = ltmp


def _mergesort(li, low, high):
    if low < high:
        mid = (low + high) // 2
        _mergesort(li, low, mid)
        _mergesort(li, mid + 1, high)
        merge(li, low, mid, high)


@cal_time
def mergesort(li):
    _mergesort(li, 0, len(li) - 1)


# 现在有一个列表，列表中的数范围都在0到100之间，列表长度大约为100万。设计算法在O(n)时间复杂度内将列表进行排序
def sort_for_100(alist):
    blist = [[] for none in range(0, 100)]
    resultlist = []
    for i in alist:
        blist[i].append(i)
    for j in blist:
        for k in j:
            resultlist.append(k)
    return resultlist


# 现在有n个数（n>10000），设计算法，按大小顺序得到前10大的数
def sort_for_10000(alist):
    blist = [0 for i in range(0, 11)]
    for i in alist:
        blist[10] = i
        for j in range(0, len(blist) - 1):
            for k in range(0, len(blist) - j - 1):
                if blist[k] < blist[k + 1]:
                    blist[k], blist[k + 1] = blist[k + 1], blist[k]
    return blist[:10]
# 最大k个数	
class Solution:
    def findKth(self , a: List[int], n: int, K: int) -> int:
        # write code here
        for i in range(K):
            for j in range(len(a)-1-i):
                if a[j] > a[j+1]:
                    a[j+1],a[j] = a[j], a[j+1]
        return a[n-K]
		
# 最小k个数
class Solution:
    def GetLeastNumbers_Solution(self , tinput: List[int], k: int) -> List[int]:
        n = len(tinput)
        for i in range(k):
            for j in range(n-1-i):
                if tinput[j] < tinput[j+1]:
                    tinput[j+1],tinput[j] = tinput[j], tinput[j+1]
        return tinput[n-k:n][::-1]
# 		
class Solution:
    def GetLeastNumbers_Solution(self, a, k):
        import heapq
        return heapq.nsmallest(k,a)

# 回文
def huiwen(a):
    a = list(a)
    leng = len(a)
    tag = True
    for i in range(leng):
        if a[i] != a[leng - 1 - i]:
            tag = False
        if i == leng - i:
            break
    return tag


# 递归
def dihui(n):
    if n == 0:
        return 0
    else:
        return n + dihui(n - 1)


# 将整数转换为任意二进制的字符串
def tostr(n, base):
    thestr = '0123456789abcdef'
    if n < base:
        return thestr[n]
    else:
        return tostr(n // base, base) + thestr[n % base]


# 处理大文件
def get_lines():
    with open('1.txt', 'r') as f:
        while True:
            data = f.readline().strip('\n')
            if not data:
                break
            yield data


if __name__ == '__main__':
    for e in get_lines():
        print(e)  # 处理每一行数据


# 将字符串 "k:1 |k1:2|k2:3|k3:4"，处理成字典 {k:1,k1:2,...}
def str2dict(str):
    dict = {}
    item1 = str.split('|')
    for i in item1:
        k, v = i.split(':')
        dict[k] = int(v)
    print(dict)


# 请按alist中元素的age由大到小排序
alist = [{'name': 'a', 'age': 20}, {'name': 'b', 'age': 30}, {'name': 'c', 'age': 25}]


def sortbyage(alist):
    return sorted(alist, key=lambda x: x['age'], reverse=True)


# 统计一个文本中单词频次最高的10个单词
def find1():
    d = {}
    with open('1.txt', 'r') as f:
        data = f.readlines()
        for i in data:
            word = i.split()
            for j in word:
                if not d.get(j):
                    d[j] = 1
                else:
                    d[j] = d[j] + 1
    d = sorted(d.items(), key=lambda x: x[1], reverse=True)
    return d


# 统计一段字符串中字符出现的次数,并倒序排列
def count_str(str):
    d = {}
    for char in str:
        d[char] = d.get(char, 0) + 1
    return sorted(d.items(), key=lambda x: x[1], reverse=True)


# 给定两个列表，怎么找出他们相同的元素和不同的元素
list1 = [1, 2, 3]
list2 = [3, 4, 5]
set1 = set(list1)
set2 = set(list2)
print(set1 & set2)
print(set1 ^ set2)


# 反转一个整数，例如-123 --> -321
def reverseint(x):
    str1 = str(x)
    if str1[0] == '-':
        str2 = str1[1:][::-1]
        int1 = - int(str2)
    else:
        str2 = str1[::-1]
        int1 = int(str2)
    return int1


# 列表解析
a = [1, 2, 3, 4, 5, 6, 7, 8]
b = [i for i in a if i > 5]


# 字符串 "-123" 转换成 -123 ，不使用内置api，例如 int()
def atoi(s):
    num = 0
    tag = False
    for v in s:
        if v == '-':
            tag = True
            continue
        for j in range(10):
            if v == str(j):
                num = num * 10 + j
    if tag:
        return -num
    else:
        return num


# 两数之和
def twoSum(nums, target):
    d = {}
    size = 0
    while size < len(nums):
        if target - nums[size] in d:
            return [d[target - nums[size]], size]
        else:
            d[nums[size]] = size
            size = size + 1
    return []


# 合并两个有序列表
def merge_list(list1, list2):
    tmp = []
    i = j = 0
    while i < len(list1) and j < len(list2):
        if list1[i] < list2[j]:
            tmp.append(list1[i])
            i = i + 1
        else:
            tmp.append(list2[j])
            j = j + 1
    return tmp + list1[i:] + list2[j:]
# 不使用tmp m,n分别是元素个数
class Solution:
    def merge(self , A, m, B, n):
        # write code here
        while(m-1>=0 and n-1>=0):
            if(A[m-1]>=B[n-1]):
                A[m+n-1] = A[m-1]
                m = m-1
            else:
                A[m+n-1] = B[n-1]
                n = n-1
        if n>=1:
            A[m:m+n] = B[:n]
        return A

# 一句话解决阶乘函数
from functools import reduce

def fun1(n):
    return reduce(lambda x, y: x * y, range(1, n + 1))


# 匹配ip
import re

re.match(r"^([0-9]{1,3}\.){3}[0-9]{1,3}$", "272.168,1,1")


# 数组中出现次数超过一半的数字
def majorityElement(alist):
    d = {}
    for i in alist:
        d[i] = d.get(i, 0) + 1
        if d[i] > len(alist) / 2:
            return i


# 求100以内的质数
num = [];
i = 2
for i in range(2, 100):
    j = 2
    for j in range(2, i):
        if (i % j == 0):
            break
    else:
        num.append(i)
print(num)

# 斐波那契数列
def fibonacci(n):
    if n == 0 or n == 1:
        return 1
    return fibonacci(n - 1) + fibonacci(n - 2)


class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None
# 反转链表
class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        cur, pre = head, None
        while cur:
            tmp = cur.next  # 暂存后继节点 cur.next
            cur.next = pre  # 修改 next 引用指向
            pre = cur       # pre 暂存 cur
            cur = tmp       # cur 访问下一节点
        return pre
link = Node(1,Node(2,Node(3,Node(4,Node(5,Node(6,Node7,Node(8.Node(9))))))))
root = reverseList(link)
while root: 
    print(roo.data)
    root = root.next

# 合并两个有序链表
class Solution:
    def mergtowlist(self, l1, l2):
        res = ListNode(0)
        if l1 is None:
            return l2
        if l2 is None:
            return l1
        if l1.var < l2.var:
            res = l1.var
            res.next = mergtowlist(l1.next, l2)
        if l1.var > l2.var:
            res = l2.var
            res.next = mergtowlist(l2.next, l1)
        return res
# 非递归
class Solution:
    def Merge(self , pHead1: ListNode, pHead2: ListNode) -> ListNode:
        # write code here
        res1 = res = ListNode(0) # res1保留头指针
        while pHead1 and pHead2:
            if pHead1.val < pHead2.val:
                res.next = pHead1
                pHead1 = pHead1.next
            else:
                res.next = pHead2
                pHead2 = pHead2.next
            res = res.next
        if not pHead1:
            res.next = pHead2
        if not pHead2:
            res.next = pHead1
        return res1.next

# 青蛙跳台阶问题
# F（n）=F（n-1）+F（n-2）
class Solution:
    def climbStairs(self,n):
        if n == 1:  
            return 1
        if n == 2:
            return 2
        return self.climbStairs(n-1) + self.climbStairs(n-2)

# 非递归 更快
class Solution:
    def jumpFloor(self , number: int) -> int:
        a,b = 1,2
        if number == 1:
            return 1
        if number == 2:
            return 2
        for _ in range(number-2):
            a, b = b, a+b 
        return b

# 字符串相加 返回字符串
class Solution:
    def addStrings(self, num1: str, num2: str) -> str:
        num1_int = 0
        num2_int = 0
        for i in num1:
            num1_int = num1_int * 10 + ord(i)-ord('0')
        for j in num2:
            num2_int = num2_int * 10 + ord(j)-ord('0')
        num_int = num1_int + num2_int
        return str(num_int)

# 用列表实现栈
class Stack():
    def __init__(self):
        self.items = []

    def push(self, x):
        self.items.append(x)

    def pop(self):
        self.items.pop()

    def isEmpty(self):
        return self.items == []

    def peek(self):
        return self.items[len(self.items) - 1]

    def size(self):
        return len(self.items)

# python列表实现队列
class Queue:
    def __init__(self):
        self.items = []

    def enqueue(self, x):
        self.items.insert(0, x)

    def dequeue(self):
        return self.items.pop()

    def isEmpty(self):
        return self.items == []

    def size(self):
        return len(self.items)


# 用两个栈 实现一个队列 list模拟栈
class MyQueue:
    def __init__(self):
        self.a = []
        self.b = [] # 实际装数据

    def push(self, x: int) -> None:
        while self.b:
            self.a.append(self.b.pop())
        self.a.append(x)
        while self.a:
            self.b.append(self.a.pop())

    def pop(self) -> int:
        return self.b.pop()

    def peek(self) -> int:
        return self.b[-1]


    def empty(self) -> bool:
        return len(self.b) == 0

# 括号匹配问题
def check_kuohao(s):
    stack = []
    for char in s:
        if char in {'(', '[', '{'}:
            stack.append(char)
        elif char == ')':
            if len(stack) > 0 and stack[-1] == '(':
                stack.pop()
            else:
                return False
        elif char == ']':
            if len(stack) > 0 and stack[-1] == '[':
                stack.pop()
            else:
                return False
        elif char == '}':
            if len(stack) > 0 and stack[-1] == '{':
                stack.pop()
            else:
                return False
    if len(stack) == 0:
        return True
    else:
        return False
# python 实现链表
class Node:
    def __init__(self, initdata):
        self.data = initdata
        self.next = None

    def getData(self):
        return self.data

    def getNext(self):
        return self.next

    def setData(self, newdata):
        self.data = newdata

    def setNext(self, newnext):
        self.next = newnext

# python 实现链表
class UnorderList:
    def __init__(self):
        self.head = None

    def isEmpty(self):
        return None == self.head

    def add(self, item):
        temp = Node(item)
        temp.setNext(self.head)
        self.head = temp

    def length(self):
        current = self.head
        count = 0
        while None != current:
            count = count + 1
            current = current.getNext()
        return count

    def search(self, x):
        current = self.head
        found = False
        while None != current and not found:
            if current.getData() == x:
                found = True
            else:
                current = current.getNext
        return found

    def remove(self, x):
        current = self.head
        previous = None  # 当前节点的上一个节点
        found = False
        while not found:
            if current.getData == x:
                found = True
            else:
                previous = current
                current = current.getNext
        if None == previous:
            self.head = current.getNext
        else:
            previous.setNext(current.getNext)

# python 实现树
class BinaryTree:
    def __init__(self, rootObj):
        self.key = rootObj
        self.leftChild = None
        self.rightChild = None

    def insertLeft(self, newnNode):
        if self.leftChild == None:
            self.leftChild = BinaryTree(newnNode)
        else:
            t = BinaryTree(newnNode)
            t.leftChild = self.leftChild
            self.leftChild = t

    def insertRight(self, newNode):
        if self.rightChild == None:
            self.rightChild = BinaryTree(newNode)
        else:
            t = BinaryTree(newNode)
            t.rightChild = self.rightChild
            self.rightChild = t

    def getRightChild(self):
        return self.rightChild

    def getLeftChild(self):
        return self.leftChild

    def setRootval(self, obj):
        self.key = obj

    def getRootval(self):
        return self.key
    # 内置前序遍历
    def preorder(self):
        print(self.key)
        if self.leftChild:
            self.leftChild.preorder()
        if self.rightChild:
            self.rightChild.preorder()

# 外置前序遍历
def preorder(tree):
    if tree:
        print(tree.getRootval)
        preorder(tree.getLeftChild())
        preorder(tree.getRightChild())

#快排完整版
class Solution:
    def sort1(self, li,lift,right):
        if right > lift:
            mid = self.sort2(li,lift,right)
            self.sort1(li,lift,mid-1)
            self.sort1(li,mid+1,right)
			
    def sort2(self, li,lift,right):
        tmp = li[lift]
        while True:
            while right > lift and li[right] >= tmp:
                right = right - 1
            li[lift] = li[right]
            while right > lift and li[lift] <= tmp:
                lift = lift + 1
            li[right] = li[lift]
            if lift == right:
                break
        li[lift] = tmp
        return lift
		
    def MySort(self , arr: List[int]) -> List[int]:
        # write code here
        self.sort1(arr,0,len(arr)-1)
        return arr
 
# 设计LRU缓存结构 函数嵌套函数
class Solution:
    def LRU(self , operators , k ):
        A,a=dict(),[]
        def set(key,value):
            if len(a)<k:
                A[key]=value
                a.append(key)
            else:
                del A[a.pop(0)]
                A[key]=value
                a.append(key)
 
        def get(key):
            if key in A.keys():
                a.remove(key)
                a.append(key)
                return A[key]
            else: return -1
 
        sol=[]
        for i in operators:
            if i[0]==1: set(i[1],i[2])
            else:
                sol.append(get(i[1]))
        return sol

		
# 实现二叉树先序，中序和后序遍历
class Solution:
    def __init__(self):
        self.firstArr = []
        self.midArr = []
        self.lastArr = []
    def threeOrders(self , root ):
        # write code here
        res = []
        res.append(self.firstSearch(root))
        res.append(self.midSearch(root))
        res.append(self.lastSearch(root))
        return res

    def firstSearch(self, root):
        if root != None:
            self.firstArr.append(root.val)
            self.firstSearch(root.left)
            self.firstSearch(root.right)
        return self.firstArr
 
    def midSearch(self, root):
        if root != None:
            self.midSearch(root.left)
            self.midArr.append(root.val)
            self.midSearch(root.right)
        return self.midArr
 
    def lastSearch(self, root):
        if root != None:
            self.lastSearch(root.left)
            self.lastSearch(root.right)
            self.lastArr.append(root.val)
        return self.lastArr


# 树的层序遍历
class Solution:
    def __init__(self):
        self.res = []
    def levelOrder(self , root: TreeNode) -> List[List[int]]:
        # write code here
        if not root:
            return []
        self.dfs(1,root)
        return self.res
     
    def dfs(self,index,r):
        if len(self.res) < index:
            self.res.append([])
        self.res[index - 1].append(r.val)
        if r.left:
            self.dfs(index + 1, r.left)
        if r.right:
            self.dfs(index + 1, r.right)
			
# 连续子数组的最大和 动态规划
class Solution:
    def FindGreatestSumOfSubArray(self , array: List[int]) -> int:
        length = len(array)
        sum_ = 0
        ret = array[0]
        for i in range(0,length):
            sum_ = max(array[i],sum_+array[i])
            ret = max(ret,sum_)
        return ret
		
# 判断链表中是否有环		
class Solution:
    def hasCycle(self , head: ListNode) -> bool:
        if not head:
            return False
        slow,fast= head,head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if slow == fast:
                return True
        return False
		
# 链表中环的入口结点
class Solution:
    def EntryNodeOfLoop(self, pHead):
        s = set()
        while pHead:
            if pHead in s:
                return pHead
            s.add(pHead)
            pHead = pHead.next
        return None	
		
if __name__ == '__main__':
    # print(sercher_sequence([1, 3, 4, 3, 2, 4, ], 3))
    # print(sercher_dichotomy([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 10))
    # print(bubble_sort([1, 7, 3, 4, 5, 6, 0, 8, 9, 10]))
    # print(select_sort([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))
    # print(insert_sort([1, 7, 3, 4, 5, 6, 0, 8, 9, 10]))
    # print(quick_sort([1, 2, 3, 4, 7, 6, 0, 8, 9, 10]))
    # print(
    #   sort_for_100([1, 7, 3, 4, 5, 3, 7, 2, 3, 4, 5, 6, 7, 8, 9, 8, 66, 55, 44, 33, 23, 43, 55, 76, 76, 0, 8, 9, 10]))
    #print(sort_for_10000([1, 2, 3, 4, 7, 6, 0, 8, 9, 10]))
    # e = [1,3,4,5]
    # e.pop(2)
    # print(e)
    pass