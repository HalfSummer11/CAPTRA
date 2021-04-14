import numpy as np
import os
import sys
import torch
from copy import deepcopy

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from metrics import rot_diff_degree, trans_diff, scale_diff
from rotations import noisy_rot_matrix, compute_rotation_matrix_from_matrix, compute_rotation_matrix_from_3d


def list_to_tree(tree):
    num_parts = len(tree)
    children = [[] for _ in range(num_parts)]
    root = None
    for p in range(num_parts):
        if tree[p] != -1:
            children[tree[p]].append(p)
        else:
            root = p
    joint_idx = (np.cumsum(np.array(tree) >= 0) - 1) * (np.array(tree) >= 0)
    return root, children, joint_idx


def convert_part_model(part):
    if isinstance(part['scale'], dict):
        num_parts = len(part['scale'])
        dim = len(part['rotation'][0].shape) - 2
        ret = {key: torch.stack([part[key][i] for i in range(num_parts)], dim=dim)
               for key in part.keys()}
        return ret
    else:
        return part


def eval_part_model(gt, pred, yaxis_only=False):
    gt = convert_part_model(gt)
    pred = convert_part_model(pred)
    sdiff = scale_diff(gt['scale'], pred['scale'])
    tdiff = trans_diff(gt['translation'], pred['translation'])
    rdiff = rot_diff_degree(gt['rotation'], pred['rotation'], yaxis_only=yaxis_only)

    diff_dict = {
        'sdiff': sdiff,
        'tdiff': tdiff,
        'rdiff': rdiff,
    }

    return diff_dict


def eval_part_full(gt, pred, per_instance=False, yaxis_only=False):
    pdiff = eval_part_model(gt, pred, yaxis_only=yaxis_only)
    pdiff.update({f'5deg5cm': torch.logical_and(pdiff['rdiff'] <= 5.0, pdiff['tdiff'] <= 0.05).float()})
    pdiff.update({f'10deg10cm': torch.logical_and(pdiff['rdiff'] <= 10.0, pdiff['tdiff'] <= 0.10).float()})
    pdiff = {f'{key}_{i}': pdiff[key][..., i] for key in pdiff for i in range(pdiff[key].shape[-1])}

    if per_instance:
        per_diff = deepcopy(pdiff)
    else:
        per_diff = {}

    pdiff = {key: torch.mean(value, dim=0) for key, value in pdiff.items()}

    return pdiff, per_diff


def part_model_batch_to_part(part, num_parts, device):  # [{'scale': [B], 'translation': [B, 3, 1], 'rotation': [B, 3, 3]} * P]
    keys = list(part[0].keys())
    dim = len(part[0]['translation'].shape) - 2
    part_model = {key: torch.stack([torch.tensor(part[p][key]) for p in range(num_parts)], dim=dim).float().to(device)
                  for key in keys}
    return convert_part_model(part_model)


def add_noise_to_part_dof(part, cfg):
    rand_type = cfg['type']  # 'uniform' or 'normal' --> we use 'normal'

    def random_tensor(base):
        if rand_type == 'uniform':
            return torch.rand_like(base) * 2.0 - 1.0
        elif rand_type == 'normal':
            return torch.randn_like(base)

    new_part = {}
    key = 'rotation'
    new_part[key] = noisy_rot_matrix(part[key], cfg[key], type=rand_type).reshape(part[key].shape)
    key = 'scale'
    new_part[key] = part[key] + random_tensor(part[key]) * cfg[key]
    key = 'translation'
    norm = random_tensor(part['scale']) * cfg[key]  # [B, P]
    direction = random_tensor(part[key].squeeze(-1))  # [B, P, 3]
    direction = direction / torch.clamp(direction.norm(dim=-1, keepdim=True), min=1e-9)  # [B, P, 3] unit vecs
    new_part[key] = part[key] + (direction * norm.unsqueeze(-1)).unsqueeze(-1)  # [B, P, 3, 1]

    return new_part


def pose_with_part(model, src):
    """
    model:
        'scale': [Bs, P]
        'rotation': [Bs, P, 3, 3]
        'translation': [Bs, P, 3, 1]
    src: [Bs, P, N, 3]
    """
    scale = model['scale']
    rotation = model['rotation']
    translation = model['translation']

    est = torch.matmul(src, rotation.transpose(-1, -2))
    est = est * scale.unsqueeze(-1).unsqueeze(-1)
    est = est + translation.transpose(-1, -2)

    return est


def reenact_with_part(recon, part):
    return pose_with_part(part, recon['points'])


def merge_reenact_canon_part_pose(part_dof, delta):  # delta: [B, P, 3, 3], [B, P, 3], [B, P, 1]
    pose = {key: value.clone() for key, value in part_dof.items()}
    if 'rotation' in delta:
        pose['rotation'] = torch.matmul(part_dof['rotation'], delta['rotation'])
    if 'scale' in delta:
        pose['scale'] = delta['scale'].squeeze(-1) * part_dof['scale']  # [B, P]
    if 'trans' in delta:
        pose['translation'] = (part_dof['translation'] +
                               part_dof['scale'].unsqueeze(-1).unsqueeze(-1) *
                               torch.matmul(part_dof['rotation'], delta['trans'].unsqueeze(-1)))
    return pose


def convert_pred_rtvec_to_matrix(pred, sym):  # pred: [B, D] or [B, P, D]
    if sym:
        return compute_rotation_matrix_from_3d(pred.reshape(-1, pred.shape[-1])).reshape(pred.shape[:-1] + (3, 3))
    else:
        return compute_rotation_matrix_from_matrix(pred.reshape(-1, 3, 3)).reshape(pred.shape[:-1] + (3, 3))


def compute_parts_delta_pose(init, final, canon):  # init, final: [B, P], canon: [B] or [B, P]
    if len(canon['scale'].shape) < len(final['scale'].shape):
        canon = {key: value.unsqueeze(1) for key, value in canon.items()}
    s_0, s_f, s_c = init['scale'], final['scale'], canon['scale']  # [B, P]
    t_0, t_f, t_c = init['translation'], final['translation'], canon['translation']  # [B, P, 3, 1]
    R_0, R_f, R_c = init['rotation'], final['rotation'], canon['rotation']  # [B, P, 3, 3]

    s_delta = s_f / s_0
    R_delta = torch.matmul(torch.matmul(R_c.transpose(-1, -2), R_f), torch.matmul(R_0.transpose(-1, -2), R_c))

    t = t_f - t_c
    if (t_0 - t_c).max() > 1e-7:
        t = t - s_delta.unsqueeze(-1).unsqueeze(-1) * torch.matmul(torch.matmul(R_f, R_0.transpose(-1, -2)), t_0 - t_c)
    t_delta = torch.matmul(R_c.transpose(-1, -2), t) / s_c.unsqueeze(-1).unsqueeze(-1)

    return {'scale': s_delta, 'rotation': R_delta, 'translation': t_delta}