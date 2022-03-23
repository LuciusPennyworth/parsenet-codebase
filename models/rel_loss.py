import torch
import numpy as np
import torch.nn as nn

def compute_vote_loss(pred, gt_vote):
    # gt_vote = gt_vote.float()

    # valid_class = (valid_mask != -1)  # remove background
    # gt_vote = gt_vote[valid_class]
    # pred = pred[valid_class]

    vote_loss = torch.sum(torch.abs(pred - gt_vote)) / (gt_vote.shape[0] * gt_vote.shape[1])  # (B,N)
    return vote_loss


def compute_normal_loss(pred, gt):

    b, N, _ = pred.shape
    normal_loss = torch.acos((pred * gt).sum(-1).clamp(-0.99, 0.99))

    normal_loss = normal_loss.sum() / (b * N)

    return normal_loss


def compute_param_loss(pred, T_gt, T_param_gt):
    '''
    only add loss to corresponding type
    pred: (B, N, 22)
    T_gt: (B, N)
    T_param_gt: (B, N, 22)
    '''
    param_list = {5: [0, 4], 1: [4, 8], 4: [8, 15], 3: [15, 22]}

    # [0, 4, 8, 15, 22]

    b, N, _ = pred.shape

    # l2_loss = nn.MSELoss(reduction='sum')
    l2_loss = nn.MSELoss()

    total_loss = 0
    length = 0
    cnt = 0
    for b in range(pred.shape[0]):
        for i in [1, 4, 5, 3]:
            index = T_gt[b] == i
            tmp_pred = pred[b][index]
            tmp_gt = T_param_gt[b][index]

            if tmp_pred.shape[0] == 0:
                continue
            if tmp_gt.sum() == 0:  # no parameters to process
                continue

            tmp_pred = tmp_pred[:, param_list[i][0]:param_list[i][1]]
            tmp_gt = tmp_gt[:, param_list[i][0]:param_list[i][1]].float()

            valid_mask = tmp_gt.sum(1) != 0

            tmp_pred = tmp_pred[valid_mask]
            tmp_gt = tmp_gt[valid_mask]

            if tmp_gt.shape[0] == 0:
                continue

            tmp_loss = l2_loss(tmp_pred, tmp_gt)

            # ignore wrong type label
            if tmp_gt.max() > 10 or tmp_loss > 50:
                continue

            total_loss += tmp_loss

            length += tmp_pred.shape[0]
            cnt += 1

    # TODO: only happened in test phase
    if cnt == 0:
        if torch.isnan(l2_loss(tmp_pred, tmp_gt.float())).sum() > 0:
            return torch.Tensor([0.0]).to(T_gt.device)
        return l2_loss(tmp_pred, tmp_gt.float())

    total_loss = total_loss / cnt

    return total_loss
