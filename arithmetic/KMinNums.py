# 最小k个数
# 输入n个整数，找出其中最小的K个数。例如输入4,5,1,6,2,7,3,8这8个数字，则最小的4个数字是1,2,3,4,

list = [4,5,1,6,2,7,3,8]

def findIndex(list,num):
    start = 0
    end = len(list) - 1
    mid = 0

    if end < 0:
        return mid

    while start<=end:
        mid = (start+end)>>1
        if list[mid] == num:
            return mid
        elif num > list[mid] :
            start = mid + 1
        else:
            end = mid - 1

    return start

def findKMins(list,k):
    res = []
    i = 0
    l = len(list)
    while i < k:
        num = list[i]
        index = findIndex(res,num)
        res.insert(index,num)
        i += 1

    while i<l:
        num = list[i]
        if res[k-1] > num:
            res.pop(k-1)
            index = findIndex(res,num)
            res.insert(index,num)
        i += 1

    return res

res = findKMins(list,4)
print(res)