# coding=utf-8

import re
import math

# 汉字
re_han = re.compile("^[\u4E00-\u9FD5]+$", re.U)

min_same_word = 1


# 计算最大子串长度
def max_substr(item, target):
    max_substr_len = 0

    m_val = str(item)
    n_val = str(target)
    m = len(m_val)
    n = len(n_val)

    res_matrix = []
    for i in range(0, m):
        res_matrix.append([])
        for j in range(0, n):
            res_matrix[i].append(0)

    for i in range(0, m):
        for j in range(0, n):
            if m_val[i] == n_val[j]:
                if i > 0 and j > 0:
                    res_matrix[i][j] = res_matrix[i - 1][j - 1] + 1
                else:
                    res_matrix[i][j] = 1
                max_substr_len = max(max_substr_len, res_matrix[i][j])

    # print(res_matrix)
    return max_substr_len


# 计算相同字数
def same_word(item, target):
    item_val = str(item)
    target_val = str(target)

    if len(item_val) > len(target_val):
        tmp = item_val
        item_val = target_val
        target_val = tmp

    cnt = 0
    for ch in item_val:
        if ch in target_val:
            cnt += 1
    return cnt


# 计算相似score
def calc_score(item, target, debug):
    final_score = 0.0

    item_len = len(item)
    target_len = len(target)

    max_len = target_len + item_len
    min_len = min(target_len, item_len)

    sub_len = max_substr(item, target)
    same_len = same_word(item, target)

    # 阀值
    max_diff_word_len = 1  # 相对于最短，最大差异个数
    min_calc_len = min(target_len, item_len)
    max_score = (min_calc_len - max_diff_word_len) / min_calc_len
    min_score = min(2 / min_calc_len, max_score)

    max_precision = math.log(sub_len + 1) / math.log(max_len + 1)
    min_precision = math.log(sub_len + 1) / math.log(min_len + 1)
    max_recall = math.log(same_len + 1) / math.log(max_len + 1)
    min_recall = math.log(same_len + 1) / math.log(min_len + 1)

    f1 = 0.0
    if max_precision + max_recall > 0:
        f1 = 2 * max_precision * max_recall / (max_precision + max_recall)

    f2 = 0.0
    if min_precision + min_recall > 0:
        f2 = 2 * min_precision * min_recall / (min_precision + min_recall)

    if debug:
        print(item + ' ' + target + ' sub_len = ' + str(sub_len))
        print(item + ' ' + target + ' same_len = ' + str(same_len))
        print(item + ' ' + target + ' f1 = ' + str(f1))
        print(item + ' ' + target + ' f2 = ' + str(f2))
        print(item + ' ' + target + ' min_score = ' + str(min_score))
        print(item + ' ' + target + ' max_score = ' + str(max_score))

    if (f1 >= min_score or f2 >= min_score):
        f = 2 * f1 * f2 / (f1 + f2)
        if debug:
            print(item + ' f = ' + str(f))

        if f > max_score:
            final_score = f
    return final_score

if __name__ == '__main__':
    calc_score('建议有什么用','有什么用',True)