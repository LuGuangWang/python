#[链表中环的入口结点] 一个链表中包含环，请找出该链表的环的入口结点。
#[解析]
#1. 一个快指针和一个慢指针，在环中，快指针肯定会反向追上慢指针。（假想两个人在操场上跑步，快的人的速度时慢的人的两倍，必将会与慢的人几次相遇）
#2. 两个指针重合的地方必定在环内，这个时候可以数一数环内有几个结点。只需要动其中一个指针遍历一遍又回到重合位置即可统计出环内结点数，假设为 k 。
#3. 这时快慢指针 pSlow 和 pFast 从链表头开始。pFast 先比 pSlow 走 k 步，
# 如果给结点编号，从数值上看有，pSlow+k = pFast，然后两个指针同步没次向前移动一步，
# 那么当 pSlow 到达环的入口节点时，pFast = pSlow+k，因为环只有 k 个节点，pFast
# 相当于又回到了环的开始即入口节点，两指针重合。

import operator as op

class Node(object):
    def __init__(self,x):
        self.val=x
        self.next=None

def fCircle(list):
    slow = list
    fast = list.next
    #快比慢，快两倍
    while fast.next is not None:
        if op.eq(slow,fast):
            break
        slow = slow.next
        fast = fast.next.next
    #计算环的长度
    k = 1
    slow = slow.next
    while slow is not fast:
        slow = slow.next
        k += 1
    #慢 + k = 快
    slow = list
    fast = list
    while k>=1:
        fast = fast.next
        k = k-1
    #两个再次相遇时即是入口
    while fast is not slow:
        slow = slow.next
        fast = fast.next

    return fast

list = Node(1)
list.next = Node(2)
list.next.next = Node(3)
list.next.next.next = Node(4)
list.next.next.next.next = Node(5)
list.next.next.next.next.next = Node(6)
list.next.next.next.next.next.next = list.next.next

dot = fCircle(list)
print(dot.val)