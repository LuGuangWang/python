import re

# lrn-gram
def getLRNGram(sentence,word,max_word_size=10):
    words = set()
    s_index = sentence.find(word)
    e_index = s_index + len(word)
    left = max_word_size - len(word)
    if s_index>=0 and left>=1:
        for i in range(left):
            l_start = max(0,s_index-i-1)
            r_end = min(len(sentence)-1,e_index+i+1)
            l_word = sentence[l_start:e_index]
            r_word = sentence[s_index:r_end]
            words.add(l_word)
            words.add(r_word)
    return words

## 单词过滤策略
def filterWordRule(oriWord):
    flag = True
    try:
        word = oriWord.strip().lower()
        ## 空串剔除
        if len(word)==0:
            flag = False
        ## 数字和字母组成的词
        elif re.match('^[a-z|0-9]+$',word):
            ## 纯字母保留
            if(re.match(r'^[a-z]+$',word)):
                flag = True
            ## 纯数字 字母与数字组合 剔除
            else:
                flag = False
        ## 数字和点组成的词剔除
        elif word.replace('.','').isdigit():
            flag = False
        else:
            flag = True
    except Exception:
        ##忽略异常
        flag = False
    return flag

## host is ip
def isIp(host):
    flag=True
    if len(host)==0 or re.match('^\d+.\d+.\d+.\d+$',host) or 'localhost' == host.lower():
        flag = False
    return flag

## 毫秒连续天数
def calc_keep_day(days):
    days.sort()
    max_keep=0
    tmp_keep=0
    for inx in range(1,len(days)):
        diff = days[inx] - days[inx-1]
        if diff<86400000:
            tmp_keep += 1
        else:
            max_keep = max(tmp_keep,max_keep)
            tmp_keep = 0
    max_keep = max(tmp_keep,max_keep)+1
    return max_keep