# coding=utf-8

# 数组中只出现一次的数字
# 在一个整数数组中，除了一个数之外，其他的数出现的次数都是两次，求出现一次的数

list = [2, 2, 3, 9, 3, 6, 6]

def findOne(list):
    res = 0
    for i in list:
        res ^= i

    return res


res = findOne(list)
print(res)