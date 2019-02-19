
# 输入两个链表，找出它们的第一个公共结点。

# 如果两个单向链表有公共的结点，也就是说两个链表从某一结点开始，它们的m_pNext都指向同一个结点。
# 但由于是单向链表的结点，每个结点只有一个m_pNext，因此从第一个公共结点开始，之后它们所有结点都是重合的，不可能再出现分叉。
# 所以，两个有公共结点而部分重合的链表，拓扑形状看起来像一个Y，而不可能像X

class Node(object):
    def __init__(self,d):
        self.val = d
        self.next = None


def findPublic(l1,l2):
    if l1 is None or l2 is None:
        return None

    pl1 = l1
    pl2 = l2

    while pl1.next and pl2.next:
        pl1 = pl1.next
        pl2 = pl2.next

    if pl1.next is None and pl2.next is None:
        print('两个链表等长')
        pl1 = l1
        pl2 = l2
    else:
        if pl1.next is None:
            pl1 = l2
        if pl2.next is None:
            pl2 = l1

        while pl1.next and pl2.next:
            pl1 = pl1.next
            pl2 = pl2.next

        if pl1.next is None:
            pl1 = l2
        if pl2.next is None:
            pl2 = l1

    while pl1.next and pl2.next:
        if pl1 == pl2:
            print(pl1.val)
            break
        pl1 = pl1.next
        pl2 = pl2.next

l1 = Node(1)
l1.next = Node(2)
l1.next.next=Node(3)
l1.next.next.next=Node(4)
l1.next.next.next.next=Node(5)

l2 = Node(0)
l2.next=l1.next.next

findPublic(l1,l2)