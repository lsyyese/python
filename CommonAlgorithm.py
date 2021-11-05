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


if __name__ == '__main__':
    print(sercher_sequence([1, 3, 4, 3, 2, 4, ], 3))
    print(sercher_dichotomy([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 10))
    print(bubble_sort([1, 0, 4, 4, 7, 6, 7, 2, 4, 10]))

