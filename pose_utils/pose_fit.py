import numpy as np
import os
import sys
import torch

from procrustes import transform_pts_mask
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


def random_choice_noreplace(l, n_sample, num_draw):
    '''
    l: 1-D array or list -> to choose from, e.g. range(N)
    n_sample: sample size for each draw
    num_draw: number of draws

    Intuition: Randomly generate numbers,
    get the index of the smallest n_sample number for each row.
    '''
    l = np.array(l)
    return l[np.argpartition(np.random.rand(num_draw, len(l)),
                             n_sample - 1,
                             axis=-1)[:, :n_sample]]


def filter_model_valid(model, valid):
    for key in ['scale', 'translation', 'rotation']:
        if key == 'scale':
            tmp = model[key]
        else:
            tmp = model[key].sum((-1, -2))
        valid_mask = torch.logical_and(torch.logical_not(torch.isnan(tmp)),
                                       torch.logical_not(torch.isinf(tmp)))
        valid = torch.logical_and(valid, valid_mask)
    return valid


def part_fit_st_no_ransac(labels, source, target, rotation, cfg, given_scale=None):  # given_scale: [B, P]
    """
    labels: [B, N], source: [B, P, N, 3], target [B, P, N, 3]
    rotation: [B, P, 3, 3]
    """
    num_parts = cfg['num_parts']
    eye = torch.cat([torch.eye(num_parts), torch.zeros(2, num_parts)], dim=0).to(labels.device)
    mask = eye[labels, ].transpose(-1, -2)  # [B, N, P] --> [B, P, N]
    valid = (mask.sum(dim=-1) > 3)

    _, scale, translation = transform_pts_mask(source, target, mask.unsqueeze(-1), mask.unsqueeze(-1),
                                               given_scale=given_scale, rotation=rotation, sym=cfg['sym'])
    model = {'rotation': rotation, 'scale': scale, 'translation': translation}  # [B, P]
    valid = filter_model_valid(model, valid)

    return model, valid
