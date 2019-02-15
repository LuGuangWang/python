#把字符串转换为整数
# '1234' -> 1234

def strToInt(strs):
    res = 0
    if strs is None:
        return None
    else:
        i = len(strs) - 1
        for s in strs:
            res += (10 ** i) * int(s)
            i = i-1
    return res

print(strToInt('1234'))