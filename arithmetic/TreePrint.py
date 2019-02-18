# 从上到下按层打印二叉树，同一层结点从左至右输出。每一层输出一行

class Tree(object):
    def __init__(self,d):
        self.val = d
        self.left = None
        self.right = None

def printTree(list):
    next = []

    for n in list:
        print(n.val,end=' ')
        if n.left is not None:
            next.append(n.left)
        if n.right is not None:
            next.append(n.right)

    list.clear()
    print('\n')

    if len(next) > 0:
        printTree(next)






root = Tree(1)
root.left = Tree(2)
root.right = Tree(3)
root.left.left = Tree(4)
root.left.right = Tree(5)
root.right.left = Tree(6)
root.right.right = Tree(7)
root.left.left.left = Tree(8)
root.left.left.right = Tree(9)



printTree([root])