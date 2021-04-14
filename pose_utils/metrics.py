import torch
import numpy as np


def rot_diff_rad(rot1, rot2, yaxis_only=False):
    if yaxis_only:
        if isinstance(rot1, np.ndarray):
            y1, y2 = rot1[..., 1], rot2[..., 1]  # [Bs, 3]
            diff = np.sum(y1 * y2, axis=-1)  # [Bs]
            diff = np.clip(diff, a_min=-1.0, a_max=1.0)
            return np.arccos(diff)
        else:
            y1, y2 = rot1[..., 1], rot2[..., 1]  # [Bs, 3]
            diff = torch.sum(y1 * y2, dim=-1)  # [Bs]
            diff = torch.clamp(diff, min=-1.0, max=1.0)
            return torch.acos(diff)
    else:
        if isinstance(rot1, np.ndarray):
            mat_diff = np.matmul(rot1, rot2.swapaxes(-1, -2))
            diff = mat_diff[..., 0, 0] + mat_diff[..., 1, 1] + mat_diff[..., 2, 2]
            diff = (diff - 1) / 2.0
            diff = np.clip(diff, a_min=-1.0, a_max=1.0)
            return np.arccos(diff)
        else:
            mat_diff = torch.matmul(rot1, rot2.transpose(-1, -2))
            diff = mat_diff[..., 0, 0] + mat_diff[..., 1, 1] + mat_diff[..., 2, 2]
            diff = (diff - 1) / 2.0
            diff = torch.clamp(diff, min=-1.0, max=1.0)
            return torch.acos(diff)


def rot_diff_degree(rot1, rot2, yaxis_only=False):
    return rot_diff_rad(rot1, rot2, yaxis_only=yaxis_only) / np.pi * 180.0


def trans_diff(trans1, trans2):  # [..., 3, 1]
    return torch.norm((trans1 - trans2).reshape((trans1 - trans2).shape[:-1]),
                      p=2, dim=-1)  # [..., 3, 1] -> [..., 3] -> [...]


def scale_diff(scale1, scale2):
    return torch.abs(scale1 - scale2)


def theta_diff(theta1, theta2):
    return torch.abs(theta1 - theta2)



