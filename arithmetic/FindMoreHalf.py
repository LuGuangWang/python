# 数组中有一个数字出现的次数超过数组长度的一半，请找出这个数字。
# 例如输入一个长度为9的数组{1,2,3,2,2,2,5,4,4}。
# 由于数字2在数组中出现了5次，超过数组长度的一半，因此输出2。如果不存在则输出0

# 数字出现的次数超过数组长度的一半，也就是说它出现的次数比其他所有数字出现的次数的和还要多

list = [1,2,3,2,2,2,5,4,4]

def findMoreHalf(list):
    res = None
    count = 0
    if list and len(list)>0:
        res = list[0]
        for n in list:
            if res == n:
                count += 1
            elif count==0:
                res = n
            else:
                count -= 1

    if res:
        count = 0
        for n in list:
            if n==res:
                count += 1
        if count <= len(list)/2:
            res = None

    return res

res = findMoreHalf(list)
print(res)



