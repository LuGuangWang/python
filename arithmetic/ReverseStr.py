#翻转单词顺序。比如一个句子：“I am a student.” 翻转以后是 “student. a am I”。
#两步完成：
#1. 对整个句子进行翻转
#2. 对单词进行翻转

def reverse(s,start,end):
    if start <= end:
        tmp = s[start]
        s[start] = s[end]
        s[end] = tmp
        reverse(s,start+1,end-1)


s = 'I am a student.'
s_list = list(s)


reverse(s_list,0,len(s)-1)
sstr = str(s_list)

print(sstr)