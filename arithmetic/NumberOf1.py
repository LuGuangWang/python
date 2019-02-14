# coding=utf-8

# 二进制中 1 的个数
# 把一个整数减去1，再和原整数做与运算，会把该整数最右边一个1变成0。
# 那么一个整数的二进制表示中有多少个1，就可以进行多少次这样的操作。

# 9表示成二进制是1001，有2位是1

def numOf1(num):
    count = 0
    while num > 0:
        count += 1
        num = num & (num - 1)
    return count

res = numOf1(9)

print(res)