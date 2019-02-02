#把字符串转换为整数
# '1234' -> 1234

def strToInt(str):
    res = 0
    if str is None:
        return None
    else:
        i = len(str) -1
        for s in str:
            res += (10 ** i) * int(s)
            i = i-1
    return res

print(strToInt(str))