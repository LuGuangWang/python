# coding=utf-8

# 一个整型数组里除了两个数字之外，其他的数字都出现了两次。请写程序找出这两个只出现一次的数字。
# 1. 进行异或，最后得到的结果 num 肯定是这两个只出现 1 次的数字（假设是a，b）的异或结果
# 2. 对于 num 中出现 1 的位，表明 a 和 b 在该位上的取值不同，即其中一个取 1，另一个取 0
# 3. 我们用一个标志为 flag，找到其中一个使得 a 和 b 不同的位（这里是从左到右的第一，假设为第 k 位）
# 4. 根据标志位 flag 来对数组进行划分，那么相同的数字肯定在第 k 位是相等了，这就保证了它们会被分到一起。
#    而对于 a 和 b，他们在第 k 位不同，最后肯定会分到不同的划分中。我们得到了两个子集，每个子集满足最考试描述的简单情况。

# 异或运算的特点：任何一个数字和自己做异或运算的结果都是0，任何数字和0运算的结果都是本身

list = [1, 2, 2, 3, 3, 4, 4, 1, 23, 43]


def sumDifTwo(list):
    res = 0
    for d in list:
        res ^= d
    return res

def findFirstBitId(res):
    id = 0
    while (res & 1) == 0:
        print('findFirstBitId res:',res)
        res = res >> 1
        id += 1
    return id

def isBitOne(num,id):
    num = num>>id
    return (num & 1)==1

res = sumDifTwo(list)
id = findFirstBitId(res)

num1 = num2 = 0
for num in list:
    if isBitOne(num,id):
        num1 ^= num
    else:
        num2 ^= num


print(res,id,num1,num2)

t = 3 ^ 4 ^ 4
print(t)