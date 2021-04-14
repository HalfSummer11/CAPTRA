import numpy as np

def subtract_mean(data):
    points_mean = np.mean(data['points'], axis=-1, keepdims=True)  # [3, N] -> [3, 1]
    data['points'] = data['points'] - points_mean
    data['meta']['points_mean'] = points_mean
    return data


def shuffle(data):
    n = data['points'].shape[-1]
    perm = np.random.permutation(n)
    for key in data.keys():
        if key in ['meta', 'nocs_corners']:
            continue
        else:
            cur_perm = perm
        data[key] = data[key][..., cur_perm]
    return data


def add_corners(data, obj_info):
    corners = np.array(obj_info['corners'])
    nocs_corners = corners[1:].copy()
    nocs_corners /= np.sqrt(np.sum((nocs_corners[:, 1:] - nocs_corners[:, :1]) ** 2, axis=-1, keepdims=True))
    nocs_corners = nocs_corners - np.mean(nocs_corners, axis=1, keepdims=True)  # [P, 2, 3]
    data['meta']['nocs_corners'] = nocs_corners
    return data


