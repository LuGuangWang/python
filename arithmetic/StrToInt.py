#把字符串转换为整数
# '-1234' -> -1234

def strToInt(strs):
    res = 0
    if strs is None:
        return None
    else:
        i = len(strs) - 1
        isNeg = False
        for s in strs:
            if(s.__eq__('-')):
                isNeg = True
            else:
                res += (10 ** i) * int(s)
            i = i-1
    if isNeg:
        res = -res
    return res

print(strToInt('-1234'))