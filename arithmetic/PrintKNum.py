# coding=utf-8

# 链表中倒数第k个结点
# （1）第一个指针从链表的头指针开始遍历向前走k-1，第二个指针保持不动；
# （2）从第k步开始，第二个指针也开始从链表的头指针开始遍历；
# （3）由于两个指针的距离保持在k-1，当第一个（走在前面的）指针到达链表的尾结点时，第二个指针（走在后面的）指针正好是倒数第k个结点。

class Node(object):
    def __init__(self,x):
        self.val = x
        self.next = None

    def sprint(self):
        tmp = self
        while tmp is not None:
            print(tmp.val)
            tmp = tmp.next


def printKNum(list,k):
    if list is None or k < 0:
        raise Exception('invalid parameters.')

    fast = slow = list
    while fast is not None and k>1:
        fast = fast.next
        k -= 1

    print('fast:',fast.val)

    if fast is None:
        return None

    while fast.next is not None:
        fast = fast.next
        slow = slow.next

    return slow



list = Node(1)
list.next=Node(2)
list.next.next=Node(3)
list.next.next.next=Node(4)
list.next.next.next.next=Node(5)
list.next.next.next.next.next=Node(6)
list.next.next.next.next.next.next=Node(7)
list.next.next.next.next.next.next.next=Node(8)
list.next.next.next.next.next.next.next.next=Node(9)

res = printKNum(list,9)

print(res.val)
#list.sprint()

