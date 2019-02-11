#输入一个递增排序的数组和一个数字S，在数组中查找两个数，是的他们的和正好是S，
import math
import sys

def search(list,num,start,end):
    if start >= end:
        return None

    mid = math.floor((start + end) / 2)
    if list[mid] == num:
        return mid
    elif list[mid] > num:
        return search(list,num,start,mid)
    else:
        return search(list,num,mid+1,end)

def searchSum(list,sum):
    res = []
    m = sys.maxsize
    l = len(list)
    for e in range(0,l):
        a = list[e]
        b = sum - a
        index = search(list,b,e,l)
        if index is not None:
            res.append((e,index))
            m = min(m,list[e]*list[index])

    return res,m

list = [1,2,3,4,5,6,7]
res = searchSum(list,7)
print('res: ',res)


