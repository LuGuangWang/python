#求子序列中的最大和
# 10 -1 8 9 -9 4


def maxSubSum(list):
    sum=lastsum=list[0]
    for index in range(1,len(list)):
        tmp = lastsum + list[index]
        if tmp>0:
            lastsum = tmp
        else:
            lastsum = list[index]
        sum = max(sum,lastsum)

    return sum

list = [30,-20,8,9,-29,4]
sum = maxSubSum(list)
print(sum)

