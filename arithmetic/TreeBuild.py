
# 重建二叉树
# 输入前序遍历序列{1,2,4,7,3,5,6,8}和中序遍历序列{4,7,2,1,5,3,8,6}，则重建二叉树并返回。
# 注意：
#已知前序和中序遍历，可以确定一棵二叉树。已知中序和后序遍历，可以确定一棵二叉树。但是，已知前序和后序遍历，不能确定一棵二叉树

# 前序遍历的第一个节点{1}一定是这棵二叉树的根节点，
# 根据中序遍历序列，可以发现中序遍历序列中节点{1}之前的{4,7,2}是这棵二叉树的左子树，{5,3,8,6}是这棵二叉树的右子树。
# 然后，对于左子树，递归地把前序子序列{2,4,7}和中序子序列{4,7,2}看成新的前序遍历和中序遍历序列。
# 此时，对于这两个序列，该子树的根节点是{2}，该子树的左子树为{4,7}、右子树为空，
# 如此递归下去（即把当前子树当做树，又根据上述步骤分析）。{5,3,8,6}这棵右子树的分析也是这样


class Tree(object):
    def __init__(self,d):
        self.val = d
        self.right = None
        self.left = None

def buildTree(preList,i,midList,start,end):
    if i >= len(preList):
        return None

    val = preList[i]
    node = Tree(val)
    index = midList.index(val)
    leftEnd = index - 1
    if start <= leftEnd:
        node.left = buildTree(preList,i+1,midList,start,leftEnd)
    rightStart = index + 1
    leftLen = index - start
    if rightStart <= end:
        node.right = buildTree(preList,i + leftLen + 1,midList,rightStart,end)

    return node



# preList = [1, 2, 4, 7, 3, 5, 6, 8]
# midList = [4, 7, 2, 1, 5, 3, 8, 6]
preList = [1, 2, 3, 4, 7]
midList = [1,2,3,7,4]
root  = buildTree(preList,0,midList,0,len(midList)-1)
root