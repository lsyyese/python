import datetime


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


# 3.冒泡排序

"""比较相邻的元素。如果第一个比第二个大，就交换他们两个。
对每一对相邻元素做同样的工作，从开始第一对到结尾的最后一对。在这一点，最后的元素应该会是最大的数。
针对所有的元素重复以上的步骤，除了最后一个。
持续每次对越来越少的元素重复上面的步骤，直到没有任何一对数字需要比较。"""


def bubble_sort(alist):
    for i in range(0, len(alist) - 1):
        for j in range(0, len(alist) - i - 1):
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


# 7.归并排序
"""归并排序主要通过先递归地将数组分为两个部分，排序后，再将元素合并到一起。
所以归根结底归并排序就是两部分组成：拆分+合并！"""


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


if __name__ == '__main__':
    print(sercher_sequence([1, 3, 4, 3, 2, 4, ], 3))
    print(sercher_dichotomy([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 10))
    print(bubble_sort([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))
    print(select_sort([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))
    print(insert_sort([1, 7, 3, 4, 5, 6, 0, 8, 9, 10]))
    print(quick_sort([1, 2, 3, 4, 7, 6, 0, 8, 9, 10]))
