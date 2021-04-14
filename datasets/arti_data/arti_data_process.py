import os
import sys
import numpy as np
from os.path import join as pjoin
import cv2

base_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(base_dir, '..'))

from datasets.data_utils import farthest_point_sample, random_point_sample, \
    get_obj2norm_pose, multiply_pose, inv_pose, pose2srt
from datasets.data_utils import get_urdf_mobility, get_model_pts, get_obj2link_dict
from misc.visualize.vis_utils import plot3d_pts


def gaussian_noise(depth, mask, sigma=0.000075, prob=0.5):
    prob_mask = np.random.uniform(size=depth.shape) < prob
    mask = np.bitwise_and(prob_mask, mask)
    std = np.random.uniform(0, sigma)
    noise = np.random.normal(0, std, size=depth.shape)
    depth[mask] += noise[mask]
    return depth


def gaussian_blur(depth, max_ksize=6, prob=0.4):
    if np.random.uniform() < prob or True:
        ksize = np.random.randint(1, max_ksize // 2 + 1)
        ksize = 2 * ksize + 1
        depth = cv2.GaussianBlur(depth, (ksize, ksize), sigmaX=0.2)
    return depth


def read_cloud(cloud_dict, num_points=4096, min_dis=2.0, is_debug=False, synthetic=False, num_parts=None, perturb=False,
               device='cuda:0'):
    camera_matrix = cloud_dict['camera_matrix']
    opengl_depth = cloud_dict['depth']
    seg = cloud_dict['seg']
    mask = opengl_depth < 1
    y, x = np.where(mask)
    near, far = cloud_dict['near'], cloud_dict['far']
    seg = seg[y, x]
    seg_max = seg.max()

    def depth2pts(depth):
        z = near * far / (far + depth * (near - far))
        permutation = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
        cam_points = (permutation @ np.matmul(np.linalg.inv(camera_matrix),
                                              np.stack([x, y, np.ones_like(x)] * z[y, x], 0))).T
        return cam_points

    cam_points = depth2pts(opengl_depth)

    if perturb:
        pert_depth = gaussian_blur(gaussian_noise(opengl_depth, mask))
        pert_points = depth2pts(pert_depth)
        dist = np.sqrt(np.sum((cam_points - pert_points) ** 2, axis=-1))
        seg[dist > 0.05] = seg_max - 1
        cam_points = pert_points

    if not synthetic:
        idx = np.where(cam_points[:, 0] < min_dis)[0]
        cam_points, seg = cam_points[idx], seg[idx]

    while len(cam_points) < num_points:
        cam_points = np.concatenate([cam_points, cam_points], axis=0)
        seg = np.concatenate([seg, seg], axis=0)
    fps_idx = farthest_point_sample(cam_points, num_points, device)
    if num_parts is not None:
        threshold = 10
        tmp_seg = seg[fps_idx]
        extra_idx = []
        for p in range(num_parts):
            num = threshold - np.count_nonzero(tmp_seg == p)
            if num > 0:
                cur_idx = np.where(seg == p)[0]
                extra_idx.append(cur_idx[random_point_sample(cur_idx, threshold)])
        if len(extra_idx) > 0:
            extra_idx = np.concatenate(extra_idx)
            fps_idx[random_point_sample(fps_idx, len(extra_idx))] = extra_idx

    cam_points, seg = cam_points[fps_idx], seg[fps_idx]

    if is_debug:
        max_label = np.max(seg) + 1
        pt_list = []
        for l in range(max_label):
            idx = np.where(seg == l)[0]
            pt_list.append(cam_points[idx])
        plot3d_pts([pt_list])

    return cam_points, seg


def generate_gt(cam_points, seg, cam2norm_dict):
    cam = np.concatenate([cam_points, np.ones_like(cam_points[..., 0:1])], axis=-1)
    num_parts = len(cam2norm_dict)
    norm = np.zeros_like(cam_points)
    pt_list = []
    for i in range(num_parts + 2):
        idx = np.where(seg == i)[0]
        if i >= num_parts:  # ground or robot
            norm[idx] = np.zeros((3, ))
        else:
            cur = cam[idx]
            cam2norm = cam2norm_dict[i]
            cur = np.matmul(cur, cam2norm.T)
            cur = cur[..., :3] / cur[..., 3:]
            norm[idx] = cur
            pt_list.append(cur)
    return norm


def base_generate_data(model_info, cam_points, seg, cam2world, link2world_dict, is_debug=False):
    obj2link_dict = model_info['obj2link']
    factors, corners = model_info['factor'], model_info['corner']
    num_parts = len(corners)
    obj2npcs = {p: get_obj2norm_pose(corners[p], factors[p]) for p in range(num_parts)}

    obj2cam_dict = multiply_pose(inv_pose(cam2world), multiply_pose(link2world_dict, obj2link_dict))
    cam2npcs = multiply_pose(obj2npcs, inv_pose(obj2cam_dict))

    npcs2cam = pose2srt(inv_pose(cam2npcs))
    npcs = generate_gt(cam_points, seg, cam2npcs)

    full_data = {'points': cam_points, 'labels': seg, 'nocs': npcs, 'nocs2camera': npcs2cam}
    return full_data


def generate_instance_info(root_dset, obj_category, item):
    urdf_src = pjoin(root_dset, 'urdf', obj_category, item)
    urdf_name = 'mobility.urdf'
    urdf_ins = get_urdf_mobility(pjoin(urdf_src, urdf_name))

    pts, norm_factors, corner_pts = get_model_pts(obj_file_list=urdf_ins['obj_name'])

    num_parts = len(urdf_ins['obj_name']) - 1

    parents = [parent - 1 for parent in urdf_ins['joint']['parent']]

    ret_dict = {'num_parts': num_parts,
                'global_corner': corner_pts[0],
                'global_factor': norm_factors[0],
                'corner': corner_pts[1:],
                'factor': norm_factors[1:],
                'obj2link': get_obj2link_dict(urdf_ins),
                'tree': parents}

    return ret_dict


