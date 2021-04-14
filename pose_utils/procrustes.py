import torch
import numpy as np
import sys

EPS = 1e-6
DEBUG = False
TIME = False


def fix_nan_and_print(info):
    def fix_nan(grad):
        # print("range = ", torch.max(grad).detach(), torch.min(grad).detach())
        if torch.any(grad != grad):
            print("!!!! nan in svd !!!!")
            return torch.zeros_like(grad)
    return fix_nan


def fix_nan(grad):
    if torch.any(grad != grad):
        print("!!!! nan in svd !!!!")
        return torch.zeros_like(grad)


def rotate_pts_batch(source, target):  # src, tgt [B, P, H, N, 3]
    M = torch.matmul(target.transpose(-1, -2), source)  # [..., 3, N] * [..., N, 3] = [..., 3, 3]
    _M = M.cpu()
    del M
    try:
        U, D, V = torch.svd(_M)  # [B, ..
    except RuntimeError:
        print("torch.svd failed to converge on ", M)
        print("source:", source)
        print("target:", target)
        sys.exit(1)

    if U.requires_grad:  # False during eval_baseline
        _M.register_hook(fix_nan)

    '''reflection!'''
    def det(A):  # A: [P, H, 3, 3]
        return torch.sum(torch.cross(A[..., 0, :], A[..., 1, :], dim=-1) * A[..., 2, :], dim=-1)

    device = source.device
    _U, _V = U.to(device), V.to(device)
    del U, V
    U, V = _U, _V
    d = det(torch.matmul(U, V.transpose(-1, -2)))
    mid = torch.zeros_like(U)
    mid[..., 0, 0] = 1.
    mid[..., 1, 1] = 1.
    mid[..., 2, 2] = d

    R = torch.matmul(torch.matmul(U, mid), V.transpose(-1, -2))

    return R


def scale_pts_batch(source, target):  # [B, P, H, N, 3]
    scale = (torch.sum(source * target, dim=(-1, -2)) /
             (torch.sum(source * source, dim=(-1, -2)) + EPS))
    return scale


def translate_pts_batch(source, target):   # [B, P, H, 3, N]
    return torch.mean((target - source), dim=-1, keepdim=True)  # [B, P, H, 3, 1]


def rot_around_yaxis_to_3d(rot_2d):
    xx, xz, zx, zz = rot_2d[..., 0, 0], rot_2d[..., 0, 1], rot_2d[..., 1, 0], rot_2d[..., 1, 1]
    yy = torch.ones_like(xx)
    zero = torch.zeros_like(xx)
    rot_3d = torch.stack([xx, zero, xz, zero, yy, zero, zx, zero, zz], dim=-1)
    rot_3d = rot_3d.reshape(rot_3d.shape[:-1] + (3, 3))
    return rot_3d


def transform_pts_batch(source, target, given_scale=None, rotation=None, sym=False):
    """
    src, tgt: [B, P, H, N, 3], given_scale: [B, P, H], rotation: [B, P, 1, 3, 3]
    """
    source_centered = source - torch.mean(source, -2, keepdim=True)
    target_centered = target - torch.mean(target, -2, keepdim=True)

    if rotation is None:
        rotation = rotate_pts_batch(source_centered, target_centered)
        if rotation is None:
            return None, None, None

    if sym:
        canon_target = torch.matmul(target, rotation)  # compare w/ pred npcs
        rot_2d, _ = transform_pts_2d_batch(source[..., [0, 2]], canon_target[..., [0, 2]])
        rot_3d = rot_around_yaxis_to_3d(rot_2d)
        rotation = torch.matmul(rotation, rot_3d)

    if given_scale is not None:
        scale = given_scale
    else:
        scale = scale_pts_batch(torch.matmul(source_centered,  # [B, P, H, N, 3]
                                             rotation.transpose(-1, -2)),  # [B, P, H, 3, 3]
                                target_centered)

    translation = translate_pts_batch(scale.reshape(scale.shape + (1, 1)) *
                                      torch.matmul(rotation, source.transpose(-1, -2)),
                                      target.transpose(-1, -2))

    return rotation, scale, translation  # [B, P, H, 3, 3], [B, P, H, 1], [B, P, H, 3, 1]


def rotate_pts_mask(source, target, w):  # src, tgt [B, P, 1, N, 3], w [B, P, H, N, 1]
    w = torch.sqrt(w + EPS)
    source = source * w  # already centered
    target = target * w
    return rotate_pts_batch(source, target)


def scale_pts_mask(source, target, w):  # [B, P, 1, N, 3], [B, P, H, N, 1]
    scale = (torch.sum(source * target * w, dim=(-1, -2)) /
             (torch.sum(source * source * w, dim=(-1, -2)) + EPS))
    return scale


def translate_pts_mask(source, target, w):  # [Bs, 3, N], [Bs, N, 1]
    w_shape = list(w.shape)
    w_shape[-2], w_shape[-1] = w_shape[-1], w_shape[-2]
    w = w.reshape(w_shape)  # [Bs, 1, N]
    w_sum = torch.clamp(torch.sum(w, dim=-1, keepdim=True), min=1.0)
    w_normalized = w / w_sum
    return torch.sum((target - source) * w_normalized, dim=-1, keepdim=True)  # [Bs, 3, 1]


def transform_pts_mask(source, target, mask, weights, given_scale=None, rotation=None, sym=False):
    """
    src, tgt [B, 1, 1, N, 3], mask [B, P, 1, N, 1], weights [B, P, H, N, 1]
    rotation [B, P, 1, 3, 3]
    """
    source_center = torch.sum(source * mask, dim=-2, keepdim=True) / torch.clamp(torch.sum(mask, dim=-2, keepdim=True), min=1.0)  #  [B, P, 1, N, 3]
    target_center = torch.sum(target * mask, dim=-2, keepdim=True) / torch.clamp(torch.sum(mask, dim=-2, keepdim=True), min=1.0)
    source_centered = (source - source_center) * mask  # [B, P, 1, N, 3]
    target_centered = (target - target_center) * mask

    if rotation is None:
        rotation = rotate_pts_mask(source_centered, target_centered, weights)
        if rotation is None:
            return None, None, None

    if sym:
        canon_target = torch.matmul(target, rotation)  # compare w/ pred npcs
        rot_2d, _ = transform_pts_2d_mask(source[..., [0, 2]], canon_target[..., [0, 2]], weights)
        rot_3d = rot_around_yaxis_to_3d(rot_2d)
        rotation = torch.matmul(rotation, rot_3d)

    if given_scale is not None:
        scale = given_scale
    else:
        scale = scale_pts_mask(torch.matmul(source_centered,  # [B, P, 1, N, 3]
                                            rotation.transpose(-1, -2)),  # [B, P, H, 3, 3]
                               target_centered, weights)
    translation = translate_pts_mask(
        scale.reshape(scale.shape + (1, 1)) * torch.matmul(rotation, source.transpose(-1, -2)),
        target.transpose(-1, -2),
        weights)

    return rotation, scale, translation


def rotate_pts_2d_batch(source, target):  # src, tgt [Bs, N, 2]
    M = torch.matmul(target.transpose(-1, -2), source)  # [Bs, 2, N] * [Bs, N, 2] = [Bs, 2, 2]
    M = M.detach()
    _M = M.cpu()
    del M

    try:
        U, D, V = torch.svd(_M)  # [P, H, ..
    except RuntimeError:
        sys.exit(1)

    if U.requires_grad:  # False during eval_baseline
        _M.register_hook(fix_nan)

    '''reflection!'''

    def det(A):  # A: [Bs, 2, 2]
        return A[..., 0, 0] * A[..., 1, 1] - A[..., 0, 1] * A[..., 1, 0]

    device = source.device
    _U, _V = U.to(device), V.to(device)
    del U, V
    U, V = _U, _V
    d = det(torch.matmul(U, V.transpose(-1, -2)))
    mid = torch.zeros_like(U)
    mid[..., 0, 0] = 1.
    mid[..., 1, 1] = d

    R = torch.matmul(torch.matmul(U, mid), V.transpose(-1, -2)).detach()

    validate = torch.matmul(R.transpose(-1, -2), R)  # [Bs, 2, 2]
    default = torch.eye(2).reshape(tuple(1 for _ in range(len(validate.shape) - 2)) + (2, 2)).float().to(validate.device)
    res = validate - default
    res = torch.abs(res).mean(dim=(-1, -2))

    valid_mask = (res < 1e-5).float().unsqueeze(-1).unsqueeze(-1)

    return valid_mask * R + (1.0 - valid_mask) * default


def rotate_pts_2d_mask(source, target, w):  # src, tgt [Bs, N, 2], w [Bs, N, 1]: a binary mask
    source = source * w  # already centered
    target = target * w
    return rotate_pts_2d_batch(source, target)


def transform_pts_2d_mask(source, target, mask):  # src, tgt [B, P, N, 2], mask [B, P, N, 1]
    source_center = torch.sum(source * mask, dim=-2, keepdim=True) / torch.clamp(torch.sum(mask, dim=-2, keepdim=True), min=1.0)  #  [Bs, N, 2]
    target_center = torch.sum(target * mask, dim=-2, keepdim=True) / torch.clamp(torch.sum(mask, dim=-2, keepdim=True), min=1.0)
    source_centered = (source - source_center) * mask  # [B, P, N, 2]
    target_centered = (target - target_center) * mask

    rotation = rotate_pts_2d_mask(source_centered, target_centered, mask)  # [B, P, 2, 2]
    if rotation is None:
        return None, None

    translation = translate_pts_mask(
        torch.matmul(rotation, source.transpose(-1, -2)),
        target.transpose(-1, -2),
        mask)

    return rotation, translation


def transform_pts_2d_batch(source, target):  # src, tgt [B, P, H, 3, 2]
    source_centered = source - torch.mean(source, -2, keepdim=True)
    target_centered = target - torch.mean(target, -2, keepdim=True)
    rotation = rotate_pts_2d_batch(source_centered, target_centered)
    if rotation is None:
        return None, None

    translation = translate_pts_batch(
        torch.matmul(rotation, source.transpose(-1, -2)),
        target.transpose(-1, -2))

    return rotation, translation


