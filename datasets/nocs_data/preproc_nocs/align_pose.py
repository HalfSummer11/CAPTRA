import numpy as np

EPS = 1e-6


def rotate_pts_batch(source, target):  # src, tgt [H, N, 3]
    M = np.matmul(target.swapaxes(-1, -2), source)  # [..., 3, N] * [..., N, 3] = [..., 3, 3]
    U, D, Vh = np.linalg.svd(M, full_matrices=True)
    d = np.linalg.det(np.matmul(U, Vh))
    mid = np.zeros_like(U)
    mid[..., 0, 0] = 1.
    mid[..., 1, 1] = 1.
    mid[..., 2, 2] = d
    R = np.matmul(np.matmul(U, mid), Vh)

    return R

def scale_pts_batch(source, target):  # [H, N, 3]
    scale = (np.sum(source * target, axis=(-1, -2)) /
             (np.sum(source * source, axis=(-1, -2)) + EPS))
    return scale


def translate_pts_batch(source, target):   # [H, 3, N]
    return np.mean((target - source), axis=-1, keepdims=True)  # [H, 3, 1]


def transform_pts_batch(source, target):  # src, tgt: [H, N, 3]
    source_centered = source - np.mean(source, -2, keepdims=True)
    target_centered = target - np.mean(target, -2, keepdims=True)
    rotation = rotate_pts_batch(source_centered, target_centered)

    scale = scale_pts_batch(np.matmul(source_centered,  # [H, N, 3]
                                      rotation.swapaxes(-1, -2)),  # [H, 3, 3]
                            target_centered)

    translation = translate_pts_batch(scale.reshape(scale.shape + (1, 1)) * np.matmul(rotation, source.swapaxes(-1, -2)),
                                      target.swapaxes(-1, -2))

    return rotation, scale, translation  # [H, 3, 3], [H, 1], [H, 3, 1]


def random_choice_noreplace(idx_range, n_sample, num_draw):
    return np.argpartition(np.random.rand(num_draw, idx_range),
                           n_sample - 1,
                           axis=-1)[:, :n_sample]


def pose_fit(source, target, num_hyps=64, inlier_th=1e-3):  # src, tgt: [N, 3]
    if len(source) < 3:
        print('len source < 3!', len(source))

    sample_idx = random_choice_noreplace(len(source), 3, num_hyps)  # [H, 3]

    src_sampled = source[sample_idx]
    tgt_sampled = target[sample_idx]
    rotation, scale, translation = transform_pts_batch(src_sampled, tgt_sampled)

    err = (target.reshape(1, -1, 3, 1)  # [N, 3] -> [1, N, 3, 1]
           - np.expand_dims(scale, (1, 2, 3))  # [H] -> [H, 1, 1, 1]
           * np.matmul(np.expand_dims(rotation, 1), source.reshape(1, -1, 3, 1))  # [H, N, 3, 3], [1, N, 3, 1]
           - np.expand_dims(translation, 1))  # [H, (1), 3, 1]
    err = err.reshape(err.shape[:-1])  # [H, N, 3]
    err = np.sqrt(np.sum(err ** 2, axis=-1))  # [H, N]
    score = (err < inlier_th).sum(axis=-1)  # [H]
    # print('err', err.mean(axis=1), err.min(axis=1), err.max(axis=1))
    # print('score out of', len(source), score)
    best_idx = np.argmax(score)

    inlier_idx = np.where(err[best_idx] < inlier_th)[0]
    if len(inlier_idx) < 3:
        return None
    src_sampled = source[inlier_idx]
    tgt_sampled = target[inlier_idx]
    rotation, scale, translation = transform_pts_batch(src_sampled, tgt_sampled)

    """
    posed_source = scale.reshape(1, 1, 1) * np.matmul(rotation.reshape(1, 3, 3), source.reshape(-1, 3, 1)) + translation.reshape(-1, 3, 1)
    posed_source = posed_source.reshape(-1, 3)

    mean = posed_source.mean(axis=0)
    plot3d_pts([[(posed_source - mean) / 200.0],
                [(target - mean) / 200.0],
                [(posed_source - mean) / 200.0, (target - mean) / 200.0]
                ])
    plot3d_pts([[posed_source - mean],
                [target - mean],
                [posed_source - mean, target - mean]
                ], limits=[[[-0.2, 0.2] for __ in range(3)] for _ in range(3)])
    """
    model = {'rotation': rotation, 'scale': scale, 'translation': translation}  # [B, P]

    return model

