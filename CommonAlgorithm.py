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
            if alist[j] < alist[min_index]:
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
        return a[n-k:n][::-1]
		
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
link = Node(1,Node(2,Node(3,Node(4,Node(5,Node(6,Node7,Node(8,Node(9))))))))
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
                current = current.getNext()
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
		
# 输入一个长度为 n 的链表，设链表中的元素的值为 ai ，返回该链表中倒数第k个节点。
class Solution:
    def FindKthToTail(self , pHead: ListNode, k: int) -> ListNode:
        # write code here
        listK = []
        if k <1:
            return None
        while pHead:
            listK.append(pHead)
            if len(listK) > k:
                listK.pop(0)
            pHead = pHead.next
        if len(listK) < k:
            return None
        else: 
            return listK[0]
#2 双指针解法
class Solution:
    def FindKthToTail(self , pHead , k ):
        first, second = pHead, pHead
        for i in range(k):
            if first == None:
                return None
            first = first.next
        while first:
            first = first.next
            second = second.next
        return second			
		
# 输入两个无环的单向链表，找出它们的第一个公共结点
class Solution:
    def FindFirstCommonNode(self , pHead1 , pHead2 ):
        setA = set()
        if pHead1 is None and pHead2 is None:
            return None
        while pHead1:
            setA.add(pHead1)
            pHead1 = pHead1.next
        while pHead2:
            if pHead2 in setA:
                return pHead2
            pHead2 = pHead2.next
        return None
		
# 2双指针解法
class Solution:
    def FindFirstCommonNode(self , pHead1 , pHead2 ):
        if pHead1 is None or pHead2 is None:
            return None
        p1, p2 = pHead1, pHead2 
        while p1 != p2:
            if p1:
                p1 = p1.next          
            else:
                p1 = pHead2
            if p2:
                p2 = p2.next
            else:
                p2 = pHead1
        return p1     
# 给定一个节点数为n的无序单链表，对其按升序排序。
class Solution:
    def sortInList(self , head):
        h = head
        l = []
        while h:
            l.append(h.val)
            h = h.next
        l.sort() 
        h = head
        i = 0      
        while h:
            h.val = l[i]
            h = h.next
            i += 1
        return head		
# 给定一个链表，请判断该链表是否为回文结构。
class Solution:
    def isPalindrome(self, head) -> bool:
        #链表为空，直接返回true
        if head is None:
            return True
 
        #找到链表的中点
        middle_point = self.middle_point(head)
        second_start = self.reverse_list(middle_point.next)
 
        #判断前半部分和后半部分是否相等
        result = True
        first = head
        second = second_start
        while result and second is not None:
            if first.val != second.val:
                result = False
            first = first.next
            second = second.next
 
        #还原链表并返回结果
        middle_point.next = self.reverse_list(second_start)
        return result
 
    #快慢指针寻找中点
    def middle_point(self, head):
        fast = head
        slow = head
        while fast.next is not None and fast.next.next is not None:
            fast = fast.next.next
            slow = slow.next
        return slow
 
    #翻转链表
    def reverse_list(self, head):
        cur, pre = head, None
        while cur:
            tmp = cur.next  # 暂存后继节点 cur.next
            cur.next = pre  # 修改 next 引用指向
            pre = cur       # pre 暂存 cur
            cur = tmp       # cur 访问下一节点
        return pre
		
# 删除给出链表中的重复元素（链表中元素从小到大有序），使链表中的所有元素都只出现一次
class Solution:
    def deleteDuplicates(self , head: ListNode) -> ListNode:
        cur = head
        if cur is None:
            return None
        while cur.next:
            if cur.val == cur.next.val:
                cur.next = cur.next.next
            else:
                cur = cur.next
        return head
# 给出一个升序排序的链表，删除链表中的所有重复出现的元素，只保留原链表中只出现一次的元素。
class Solution:
    def deleteDuplicates(self, head):
        if not head:
            return head
        dummy = ListNode(0)
        dummy.next=head

        cur = dummy
        while cur.next and cur.next.next:
            if cur.next.val == cur.next.next.val:
                data = cur.next.val
                while cur.next and cur.next.val == data:
                    cur.next = cur.next.next
            else:
                cur = cur.next
        return dummy.next
#旋转数组的最小数字
class Solution:
    def minNumberInRotateArray(self , rotateArray: List[int]) -> int:
        left = 0
        right = len(rotateArray)-1
        while left < right:
            mid = (left+right)//2
            if rotateArray[mid] > rotateArray[right]:
                left = mid + 1
            elif rotateArray[mid] < rotateArray[right]:
                right = mid
            else:
                right -= 1
        return rotateArray[left]
        # write code here
		
# 二维数组中的查找
class Solution:
    def Find(self , target: int, array: List[List[int]]) -> bool:
        rows = len(array)
        nums = len(array[0])
        if rows == 0:
            return False
        lift ,down = rows -1, 0
        while lift >=0 and down <= nums -1:
            if array[lift][down] == target:
                return True
            elif array[lift][down] > target:
                lift = lift - 1
            else:
                down = down + 1
        return False
# 寻找峰值
class Solution:
    def findPeakElement(self , nums: List[int]) -> int:
        lennums = len(nums)
        if max(nums) == nums[0]:
            return 0
        if max(nums) == nums[lennums -1]:
            return lennums -1
        for i in range(1,lennums):
            if nums[i] > nums[i-1] and nums[i] > nums[i+1]:
                return i
# 前序遍历
class Solution:
    def __init__(self):
        self.array = []
    def preorderTraversal(self , root: TreeNode) -> List[int]:
        if root:
            self.array.append(root.val)
            self.preorderTraversal(root.left)
            self.preorderTraversal(root.right)
        return self.array
# 二叉树的最大深度
class Solution:
    def maxDepth(self , root: TreeNode) -> int:
        if root is None:
            return 0
        if root.left is None and root.right is None:
            return 1
        l = self.maxDepth(root.left)
        r = self.maxDepth(root.right)
        return l+1 if l > r else r+1

# 二叉树中和为某一值的路径(一)
class Solution:
    def hasPathSum(self , root: TreeNode, sum: int) -> bool:
        if root is None:
            return False
        if root.right is None and root.left is None and root.val==sum:
            return True
        return self.hasPathSum(root.left,sum-root.val) or self.hasPathSum(root.right,sum-root.val)
		
# 二叉搜索树与双向链表
# 先中序遍历，将所有的节点保存到一个列表中。对这个list[:-1]进行遍历，
# 每个节点的right设为下一个节点，下一个节点的left设为上一个节点。
class Solution:
    def Convert(self , pRootOfTree ):
        # write code here
        if not pRootOfTree: return None
        self.arr = []
        self.midTraversal(pRootOfTree)
        for i,v in enumerate(self.arr[:-1]):
            v.right = self.arr[i+1]
            self.arr[i+1].left = v
        return self.arr[0]
    def midTraversal(self,root):
        if not root: return
        self.midTraversal(root.left)
        self.arr.append(root)
        self.midTraversal(root.right)
# 对称的二叉树
class Solution:
    def isBanceTree(self,leftTree,rightTree):
        if leftTree == None and rightTree == None:
            return True
        elif leftTree != None and rightTree == None:
            return False
        elif leftTree == None and rightTree != None:
            return False
        elif leftTree.val != rightTree.val:
            return False
        else:
            return self.isBanceTree(leftTree.left, rightTree.right) and self.isBanceTree(leftTree.right, rightTree.left)
    def isSymmetrical(self , pRoot: TreeNode) -> bool:
        # write code here
        if pRoot == None:
            return True
        return self.isBanceTree(pRoot.left, pRoot.right)
		
# 判断是不是平衡二叉树
class Solution:
    def deep(self,pRoot):
        if not pRoot:
            return 0
        L=self.deep(pRoot.left)
        R=self.deep(pRoot.right)
        return max(L,R)+1
    def IsBalanced_Solution(self, pRoot):
        # write code here
        if not pRoot:
            return True
        L=self.deep(pRoot.left)
        R=self.deep(pRoot.right)
        if abs(L-R)>1:
            return False
        else:
            return self.IsBalanced_Solution(pRoot.left) and self.IsBalanced_Solution(pRoot.right)
# 合并二叉树
class Solution:
    def mergeTrees(self , t1: TreeNode, t2: TreeNode) -> TreeNode:
        if t1 is None:
            return t2
        if t2 is None:
            return t1
        t1.val = t1.val + t2.val
        t1.left = mergeTrees(t1.left,t2.left)
        t1.right = mergeTrees(t1.right,t2.right)
        return t1
# 二叉树的镜像
class Solution:
    def Mirror(self , pRoot: TreeNode) -> TreeNode:
        if pRoot is None:
            return None
        pRoot.left ,pRoot.right = pRoot.right,pRoot.left
        
        pRoot.left = self.Mirror(pRoot.left)
        pRoot.right = self.Mirror(pRoot.right)
        return pRoot
# 判断是不是二叉搜索树
class Solution:
    def isValidBST(self , root: TreeNode) -> bool:
        if root is None:
            return None
        if root.left is None and root.right is None:
            return True
        if root.left and root.right:
            if root.left.val < root.val and root.right.val > root.val:
                return True
            else:
                return False
        if root.left is None and root.right:   
            if root.right.val > root.val:
                return True
            else:
                return False
        if root.left and root.right is None:
            if root.left.val < root.val:
                return True
            else:
                return False
        return self.isValidBST(root.left) and self.isValidBST(root.right)
# 判断是不是完全二叉树
class Solution:
    def isCompleteTree(self , root: TreeNode) -> bool:
        # write code here
        if not root:
            return True
        listT = [root]
        index = 0
        while index < len(listT):
            if listT[index] is not None:
                listT.append(listT[index].left)
                listT.append(listT[index].right)
            index +=1
        while listT[-1] is None:
            listT.pop()
        for q in listT:
            if q is None:
                return False
        return True
		
# 重建二叉树
class Solution:
    def reConstructBinaryTree(self , pre: List[int], vin: List[int]) -> TreeNode:
      # write code here
      if not pre or not vin:
          return None
      if len(pre) == 1:
          return TreeNode(pre[0])
 
      # 前序遍历的第一个元素是根节点
      root_val = pre[0]
      root = TreeNode(root_val)
      # 从中序遍历中找到根节点位置
      root_idx = vin.index(root_val)
      # 根节点左侧元素为左子树，右侧元素为右子树
      left_num = root_idx
      right_num = len(vin) - root_idx - 1
      # 递归构建左右子树
      root.left = self.reConstructBinaryTree(pre[1: left_num + 1], vin[:root_idx])
      root.right = self.reConstructBinaryTree(pre[-right_num:], vin[root_idx+1:])
#最长公共前缀
class Solution:
    def longestCommonPrefix(self , strs ):
        # write code here
        if len(strs)==0 or strs=="":
            return ""
        pre = strs[0]
        i=1
        while i<len(strs):
            while pre != strs[i][:len(pre)]:
                pre = pre[:(len(pre)-1)]
            i+=1
        return pre

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