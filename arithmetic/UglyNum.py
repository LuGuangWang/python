# 我们把只包含因子2、3和5的数称作丑数（Ugly Number）。
# 求按从小到大的顺序的第1500个丑数。例如6、8都是丑数，但14不是，因为它包含因子7。习惯上我们把1当做第一个丑数。
# 新的丑数只能是原来丑数的 2，3，或 5 倍，我们只需要记住上次更新后最后的因子位置即可

def findUglyNum(index):
    res = [1]
    i = m2 = m3 = m5 = 0
    while i < index-1:
        M2 = res[m2] * 2
        M3 = res[m3] * 3
        M5 = res[m5] * 5
        next = min(M2,M3,M5)
        res.append(next)

        if M2 <= next:
            m2 += 1

        if M3 <= next:
            m3 += 1

        if M5 <= next:
            m5 += 1

        i += 1
    return res.pop()

res = findUglyNum(1500)
print(res)
