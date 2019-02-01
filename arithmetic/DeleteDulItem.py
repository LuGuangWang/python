#[删除链表中重复的结点]
#在一个排序的链表中，存在重复的结点，请删除该链表中重复的结点，重复的结点不保留，返回链表头指针。
#例如，链表1->1->2->3->3->4->4->5->5
#处理后为
#2->5
class Node(object):

    def __init__(self,x):
        self.val = x
        self.next = None

    def sprint(self):
        tmp = self
        while tmp is not None:
            print(tmp.val)
            tmp = tmp.next
        print('\n')

def delDulp(list):
    head=last=None
    tmp = list
    while tmp is not None:
        flag = False
        next = tmp.next
        #要考虑2->2->3->3 , 1->6->6的情况
        while (next is not None):
            if (tmp.val == next.val):
                next = next.next
                flag = True
            elif flag:
                tmp = next
                next = tmp.next
                flag = False
            else:
                break
        if flag:
            tmp = next

        if head is None:
            head = tmp
            last = tmp
        else:
            last.next = tmp
            last = last.next

        tmp=next
    return head

#1->1->2->3->3->4->4->5->6->6
list = Node(1)
list.next=Node(2)
list.next.next=Node(2)
list.next.next.next=Node(2)
list.next.next.next.next=Node(3)
list.next.next.next.next.next=Node(4)
list.next.next.next.next.next.next=Node(4)
list.next.next.next.next.next.next.next=Node(5)
list.next.next.next.next.next.next.next.next=Node(6)
list.next.next.next.next.next.next.next.next.next=Node(6)

res = delDulp(list)
res.sprint()
