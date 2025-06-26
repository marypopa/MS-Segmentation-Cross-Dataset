import torch

def _take_channels(*xs, ignore_channels=None):
    if ignore_channels is None:
        return xs
    else:
        channels = [channel for channel in range(xs[0].shape[1]) if channel not in ignore_channels]
        xs = [torch.index_select(x, dim=1, index=torch.tensor(channels).to(x.device)) for x in xs]
        return xs


def _threshold(x, threshold=None):
    if threshold is not None:
        return (x > threshold).type(x.dtype)
    else:
        return x


def iou(pr, gt, eps=1e-7, threshold=None, ignore_channels=None):
    """Calculate Intersection over Union between ground truth and prediction
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: IoU (Jaccard) score --> for each batch element!
    """

    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)

    intersection = torch.sum(gt * pr,dim=(1,2,3))  #sum over C,H,W of a (N,C,H,W) tensor -> the result is a vector of the length of the batch size
    union = torch.sum(gt,dim=(1,2,3)) + torch.sum(pr,dim=(1,2,3)) - intersection + eps
    iou_functional = (intersection + eps) / union

    return iou_functional

jaccard = iou


def f_score(pr, gt, beta=1, eps=1e-7, threshold=None, ignore_channels=None):
    """Calculate F-score between ground truth and prediction
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        beta (float): positive constant
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: F score
    """

    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)

    tp = torch.sum(gt * pr,dim=(1,2,3))
    fp = torch.sum(pr,dim=(1,2,3)) - tp
    fn = torch.sum(gt,dim=(1,2,3)) - tp

    # score = ((1 + beta**2) * tp + eps) / ((1 + beta**2) * tp + beta**2 * fn + fp + eps)
    score = ((1 + beta**2) * tp + eps) / (beta**2 * torch.sum(gt,dim=(1,2,3)) + torch.sum(pr,dim=(1,2,3)) + eps)

    return score


def accuracy(pr, gt, threshold=0.5, ignore_channels=None):
    """Calculate accuracy score between ground truth and prediction
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: precision score
    """
    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)

    tp = torch.sum(gt == pr,dim=(1,2,3), dtype=pr.dtype)
    score = tp / gt.view(gt.size(0), -1).size(1)
    # score = tp / gt.view(, -1).shape[0]

    return score


def precision(pr, gt, eps=1e-7, threshold=None, ignore_channels=None):
    """Calculate precision score between ground truth and prediction
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: precision score
    """

    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)

    tp = torch.sum(gt * pr,dim=(1,2,3))
    fp = torch.sum(pr,dim=(1,2,3)) - tp

    score = (tp + eps) / (tp + fp + eps)

    return score


def recall(pr, gt, eps=1e-7, threshold=None, ignore_channels=None):
    """Calculate Recall between ground truth and prediction
    Args:
        pr (torch.Tensor): A list of predicted elements
        gt (torch.Tensor):  A list of elements that are to be predicted
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: recall score
    """

    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)

    tp = torch.sum(gt * pr,dim=(1,2,3))
    fn = torch.sum(gt,dim=(1,2,3)) - tp

    score = (tp + eps) / (tp + fn + eps)

    return score

def fp(pr, gt, eps=1e-7, threshold=None, ignore_channels=None):

    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)

    tp = torch.sum(gt * pr,dim=(1,2,3))
    fp = torch.sum(pr,dim=(1,2,3)) - tp

    return fp

def fn(pr, gt, eps=1e-7, threshold=None, ignore_channels=None):

    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)

    tp = torch.sum(gt * pr,dim=(1,2,3))
    fn = torch.sum(gt,dim=(1,2,3)) - tp

    return fn

def tp(pr, gt, eps=1e-7, threshold=None, ignore_channels=None):

    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)

    tp = torch.sum(gt * pr,dim=(1,2,3))

    return tp

