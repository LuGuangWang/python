
# 旋转数组的最小数
# 把一个数组最开始的若干个元素搬到数组的末尾，我们称之为数组的旋转
# 数组{3,4,5,1,2}为{1,2,3,4,5}的一个旋转，该数组的最小值为1。
# 思路：二分查找的变种
# 两个指针分别指向数组的第一个元素和最后一个元素。
# 如果中间元素位于前面的递增子数组，那么它应该大于或者等于第一个指针指向的元素
# 如果中间元素位于后面的递增子数组，那么它应该小于或者等于第二个指针指向的元素

list = [5,6,7,2,3,4,5]

def findMin(list,start,end):
    if list is None or start<0 or end < 0:
        return 'parameter error.'

    if start + 1 >= end:
        return list[end]
    res = 0
    mid = round((start + end)/2)
    # 注意首尾相等的特殊情况
    if list[start] == list[end]:
        res = findMin(list,start+1,end - 1)
    elif list[mid]>=list[start]:
        res = findMin(list,mid,end)
    else :
        res = findMin(list,start,mid)

    return res

res = findMin(list,0,len(list)-1)
print(res)
