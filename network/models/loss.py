import torch
import os
import sys
from os.path import join as pjoin
BASEPATH = os.path.dirname(__file__)
sys.path.insert(0, pjoin(BASEPATH, '..', '..'))
from pose_utils.rotations import matrix_to_rotvec

EPS = 1e-6


def vector_loss(x, loss='l2'):  # x: [..., D] -> [...]
    """ Follows the practice of Xiaolong's code"""
    if loss == 'l2':  # it's "MSELoss" in the paper, but it's implemented as l2
        return torch.norm(x, p=2, dim=-1)
    elif loss == 'l1':
        return torch.norm(x, p=1, dim=-1)
    else:
        assert 0, "Unsupported loss type {}".format(loss)


def choose_coord_by_label(x, labels, last_dim=3):
    """
    labels = None, x = [B, N, 3] or
    labels = [B, N] in [0, P - 1], x = [B, N, last_dim * P]
    """
    if labels is None:
        return x

    old_shape = x.shape
    num_parts = int(old_shape[-1] // last_dim)
    new_shape = old_shape[:-1] + (num_parts, last_dim)
    x = x.reshape(new_shape)  # [B, N, P, D]
    y = torch.cat([x, torch.zeros_like(x[..., :2, :])], dim=-2)

    B_range = torch.tensor(range(labels.shape[0])).reshape(-1, 1)  # [B, 1]
    N_range = torch.tensor(range(labels.shape[1])).reshape(1, -1)  # [1, N]
    z = y[B_range, N_range, labels]
    return z


def compute_nocs_loss(nocs_per_part, nocs_gt, labels=None, confidence=None, loss='l2', self_supervise=True,
                      per_instance=False, sym=False, pwm_num=128):
    """
    *Need to deal with seg >= num_parts
    nocs_per_part: [(B,) 3 * P, N] or [(B,), 3, N]
    nocs_gt: [(B,) 3, N]
    labels: [(B,) N] or None
    confidence: [(B,) N]
    """
    nocs_per_part = nocs_per_part.transpose(-1, -2)  # [(B,) N, 3 * P] or [(B,) N, 3]
    nocs_gt = nocs_gt.transpose(-1, -2)  # [(B,) N, 3]

    if confidence is None or not self_supervise:  # not self-supervising
        conf = torch.ones(nocs_gt.shape[:-1]).to(nocs_gt.device)
    else:
        conf = confidence

    if labels is not None and nocs_per_part.shape[-1] > 3:
        nocs_pred = choose_coord_by_label(nocs_per_part, labels, last_dim=3)
        num_parts = int(nocs_per_part.shape[-1] // 3)
        mask = labels < num_parts
    else:
        nocs_pred = nocs_per_part
        mask = None

    if sym:
        return compute_sym_nocs_loss(nocs_pred, nocs_gt, labels, pwm_num=pwm_num)

    diff = nocs_pred - nocs_gt  # [B, N, 3]
    raw = vector_loss(diff, loss=loss)
    raw = raw * conf
    if mask is None:
        ret = torch.mean(raw)
    else:
        ret = torch.sum(raw * mask) / max(torch.sum(mask), 1.0)
    ret -= 0.1 * torch.mean(torch.log(conf))
    if per_instance:
        return ret, raw
    else:
        return ret


def compute_sym_nocs_loss(nocs_pred, nocs_gt, labels, pwm_num=128):  # [B, N, 3], [B, N]
    x_gt, y_gt, z_gt = nocs_gt[..., 0], nocs_gt[..., 1], nocs_gt[..., 2]
    x_pred, y_pred, z_pred = nocs_pred[..., 0], nocs_pred[..., 1], nocs_pred[..., 2]
    dist = torch.sqrt((y_gt - y_pred) ** 2 + torch.abs(x_gt ** 2 + z_gt ** 2 - x_pred ** 2 - z_pred ** 2) + 1e-8)

    mask = labels == 0  # [B, N]
    valid_mask = (torch.sum(mask, dim=-1) > 0).float()  # [B]
    dist_loss = torch.sum(dist * mask) / torch.clamp(torch.sum(mask), min=1.0)

    # only care about where labels == 0
    batch_size = len(labels)
    idxs = []
    for b in range(batch_size):
        idx = torch.where(labels[b] == 0)[0]  # [N]
        if len(idx) == 0:
            idx = torch.where(labels[b] == 1)[0]
        sample = torch.randint(len(idx), (pwm_num, ))  # [M]
        sample_idx = idx[sample]
        # plot3d_pts([[nocs_gt[b][sample_idx].detach()], [nocs_pred[b][sample_idx].detach()]])
        idxs.append(sample_idx)
    idxs = torch.stack(idxs, dim=0)  # [B, M] in [N]
    batch_idx = torch.tensor(range(batch_size)).long().reshape(batch_size, 1)  # [B, 1] --> [B, M] in [B]

    sampled_gt = nocs_gt[batch_idx, idxs]  # [B, M, 3]
    sampled_pred = nocs_pred[batch_idx, idxs]  # [B, M, 3]
    sampled_labels = labels[batch_idx, idxs]

    def dist_mat(pts):  # [Bs, N, 3]
        diff = pts.unsqueeze(-2) - pts.unsqueeze(-3)  # [Bs, N, 1, 3] - [Bs, 1, N, 3]
        diff_norm = torch.norm(diff, p=2, dim=-1)  # [Bs, N, N]
        return diff_norm

    pwm_diff = torch.abs(dist_mat(sampled_gt) - dist_mat(sampled_pred)).mean(dim=(-1, -2))  # [Bs, N, N]
    pwm_diff = torch.sum(pwm_diff * valid_mask) / torch.clamp(torch.sum(valid_mask), min=1.0)

    return dist_loss, pwm_diff


def compute_miou_loss(pred, labels, per_instance=False):  # pred: [(B,) P, N], labels: [(B,) N]
    pred = pred.transpose(-1, -2)  # [(B,) N, P]
    C = pred.shape[-1]
    gt = torch.eye(C)[labels, ].to(labels.device)
    I = torch.sum(pred * gt, dim=-2)  # [(B, ), N, P] -> [(B,) P]
    U = torch.sum(pred + gt, dim=-2) - I  # [(B, ), N, P] -> [(B,) P]
    mIoU = I / (U + EPS)  # per point cloud, per part
    loss = 1.0 - torch.mean(mIoU)

    if per_instance:
        return loss, mIoU
    else:
        return loss


def compute_hard_miou_loss(pred, gt, num_parts, per_instance=False):
    gt = torch.eye(num_parts)[gt,].to(gt.device)
    pred = torch.eye(num_parts)[pred,].to(pred.device)
    I = torch.sum(pred * gt, dim=-2)  # [(B, ), N, P] -> [(B,) P]
    U = torch.sum(pred + gt, dim=-2) - I  # [(B, ), N, P] -> [(B,) P]
    mIoU = I / (U + EPS)  # per point cloud, per part
    loss = 1.0 - torch.mean(mIoU)

    if per_instance:
        return loss, mIoU
    else:
        return loss


def rot_trace_loss(rot1, rot2, metric='l1'):
    '''||trace(R1*R2^T) - 3||^2'''
    if 'exp' in metric:
        exp1 = matrix_to_rotvec(rot1)
        exp2 = matrix_to_rotvec(rot2)
        diff = exp1 - exp2

        if metric == 'exp_l2':
            return diff ** 2
        elif metric == 'exp_l1':
            return torch.abs(diff)
        else:
            assert 0, f'Unsupported metric {metric}'
    elif metric == 'frob':
        mat_diff = rot1 - rot2
        tmp = torch.matmul(mat_diff, mat_diff.transpose(-1, -2))
        trace = tmp[..., 0, 0] + tmp[..., 1, 1] + tmp[..., 2, 2]
        return trace
    else:
        mat_diff = torch.matmul(rot1, rot2.transpose(-1, -2))   # [Bs, 3, 3]
        diff = mat_diff[..., 0, 0] + mat_diff[..., 1, 1] + mat_diff[..., 2, 2]  # [B, 1]
        if metric == 'l2':
            return ((diff - 3.)) ** 2
        elif metric == 'l1':
            return torch.abs(diff - 3)
        else:
            assert 0, f'Unsupported metric {metric}'


def rot_yaxis_loss(rot1, rot2, metric='l2'):
    y1, y2 = rot1[..., 1], rot2[..., 1]
    diff = y1 - y2
    if metric == 'l2':
        return (diff ** 2).sum(-1)
    elif metric == 'l1':
        return torch.norm(diff, p=2, dim=-1)
    else:
        assert 0, f'Unsupported metric {metric}'


def trans_loss(trans1, trans2, metric='l1'):
    if metric == 'l2':
        return torch.sum(((trans1 - trans2)) ** 2, dim=(-1, -2))
    elif metric == 'l1':
        return torch.norm((trans1 - trans2).reshape((trans1 - trans2).shape[:-1]), p=2,
                          dim=-1)  # [B, P, H, 3, 1] -> [B, P, H, 3] -> [B, P, H]
    else:
        assert 0, f'Unsupported metric {metric}'


def scale_loss(scale1, scale2, metric='l1'):
    if metric == 'l2':
        return ((scale1 - scale2)) ** 2
    elif metric == 'l1':
        return torch.abs(scale1 - scale2)  # [B, P, H]
    else:
        assert 0, f'Unsupported metric {metric}'


def compute_point_pose_loss(gt_pose, pred_pose, pts, metric='l1'):
    from part_dof_utils import pose_with_part
    gt_pts = pose_with_part(gt_pose, pts)   # [B, P, N, 3]
    pred_pts = pose_with_part(pred_pose, pts)
    pts_diff = (gt_pts - pred_pts)
    if metric == 'l2':
        dist = torch.sum(pts_diff ** 2, dim=-1)  # [B, P, N]
    elif metric == 'l1':
        dist = torch.norm(pts_diff, p=2, dim=-1)  # [B, P, N]
    else:
        assert 0, f'Unsupported metric {metric}'
    return dist.mean(), dist


def compute_part_dof_loss(gt, pred, pose_loss_type, collapse=True):
    sloss = scale_loss(gt['scale'], pred['scale'], metric=pose_loss_type['s'])
    tloss = trans_loss(gt['translation'], pred['translation'], metric=pose_loss_type['t'])
    rloss = rot_trace_loss(gt['rotation'], pred['rotation'], metric=pose_loss_type['r'])

    loss_dict = {
        'sloss': sloss,
        'tloss': tloss,
        'rloss': rloss,
    }
    if collapse:
        loss_dict = {key: value.mean() for key, value in loss_dict.items()}
    return loss_dict


if __name__ == '__main__':

    a = torch.randn(2, 4, 5, 3)
    b = torch.randn(2, 5, 3)
    c = torch.randint(0, 4, (2, 5))

