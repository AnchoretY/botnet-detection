"""
    评估二分类结果的函数
`pred`和`target`为相同长度的`torch.LongTensor`或像相同长度的numpy.ndarray
"""


def accuracy(pred, target):
    """
        所有样本中分类正确的样本所占比例
        acc = (TP+TN)/(TP+TN+FP+FN)
    """
    return (pred == target).sum().item() / len(target)


def true_positive(pred, target):
    """
        被正确预测为正样本的条数
    """
    return (target[pred == 1] == 1).sum().item()


def false_positive(pred, target):
    """
        被误判为正样本的条数
    """
    return (target[pred == 1] == 0).sum().item()


def true_negative(pred, target):
    """
        被正确预测为负样本的条数
    """
    return (target[pred == 0] == 0).sum().item()


def false_negative(pred, target):
    """
        被误判为负样本的条数
    """
    return (target[pred == 0] == 1).sum().item()


def recall(pred, target):
    """
        正确预测为正样本的占全部正样本的比例
    """
    try:
        return true_positive(pred, target) / (target == 1).sum().item()
    except:  # divide by zero
        return -1


def precision(pred, target):
    """
        预测为正样本的样本中正确预测的比例
    """
    try:
        prec = true_positive(pred, target) / (pred == 1).sum().item()
        return prec
    except:  # divide by zero
        return -1


def f1_score(pred, target):
    """
        f1-score = 2*precision*recall/(precision+recall)
    """
    prec = precision(pred, target)
    rec = recall(pred, target)
    try:
        return 2 * (prec * rec) / (prec + rec)
    except:
        return 0


def false_positive_rate(pred, target):
    """
        负样本中被判为正样本的比例
    """
    try:
        return false_positive(pred, target) / (target == 0).sum().item()
    except:  # divide by zero
        return -1


def false_negative_rate(pred, target):
    """
    正样本中被误判为负样本的比例，1 - recall/true_positive_rate
    """
    try:
        return false_negative(pred, target) / (target == 1).sum().item()
    except:  # divide by zero
        return -1
