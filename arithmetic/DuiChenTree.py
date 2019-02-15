# 是否是对称二叉树

class Tree(object):
    def __init__(self,d):
        self.val = d
        self.right = None
        self.left = None

def isDuiChen(n1, n2):
    if n1 is None and n2 is None:
        return True
    elif n1 is None or n2 is None:
        return False

    if n1.val != n2.val:
        return False

    return isDuiChen(n1.left,n2.right) and isDuiChen(n1.right,n2.left)

root = Tree(1)
right = Tree(2)
left = Tree(2)
root.left = left
root.right = right
left.left = Tree(3)
left.right = Tree(4)
right.left=Tree(4)
right.right=Tree(3)

if root is None:
    res = True

res = isDuiChen(root.left,root.right)

print(res)
