import numpy as np
import torch
import os
import sys
from os.path import join as pjoin
BASEPATH = os.path.dirname(__file__)
sys.path.insert(0, pjoin(BASEPATH, '..', '..'))
from pose_utils.part_dof_utils import convert_part_model, pose_with_part


def pts_inside_box(pts, bbox):
    # pts: N x 3
    # bbox: 8 x 3 (-1, 1, 1), (1, 1, 1), (1, -1, 1), (-1, -1, 1), (-1, 1, -1), (1, 1, -1), (1, -1, -1), (-1, -1, -1)
    u1 = bbox[5, :] - bbox[4, :]
    u2 = bbox[7, :] - bbox[4, :]
    u3 = bbox[0, :] - bbox[4, :]

    up = pts - np.reshape(bbox[4, :], (1, 3))
    p1 = np.matmul(up, u1.reshape((3, 1)))
    p2 = np.matmul(up, u2.reshape((3, 1)))
    p3 = np.matmul(up, u3.reshape((3, 1)))
    p1 = np.logical_and(p1 > 0, p1 < np.dot(u1, u1))
    p2 = np.logical_and(p2 > 0, p2 < np.dot(u2, u2))
    p3 = np.logical_and(p3 > 0, p3 < np.dot(u3, u3))
    return np.logical_and(np.logical_and(p1, p2), p3)


def iou_3d(bbox1, bbox2, nres=50):
    bmin = np.min(np.concatenate((bbox1, bbox2), 0), 0)
    bmax = np.max(np.concatenate((bbox1, bbox2), 0), 0)
    xs = np.tile(np.linspace(bmin[0], bmax[0], nres).reshape(-1, 1, 1), (1, nres, nres))
    ys = np.tile(np.linspace(bmin[1], bmax[1], nres).reshape(1, -1, 1), (nres, 1, nres))
    zs = np.tile(np.linspace(bmin[2], bmax[2], nres).reshape(1, 1, -1), (nres, nres, 1))
    pts = np.stack([xs, ys, zs], axis=-1)
    flag1 = pts_inside_box(pts, bbox1)
    flag2 = pts_inside_box(pts, bbox2)
    intersect = np.sum(np.logical_and(flag1, flag2))
    union = np.sum(np.logical_or(flag1, flag2))
    if union == 0:
        return 1
    else:
        return intersect / float(union)


def nocs_iou_3d(bbox_3d_1, bbox_3d_2):
    bbox_1_max = np.amax(bbox_3d_1, axis=0)
    bbox_1_min = np.amin(bbox_3d_1, axis=0)
    bbox_2_max = np.amax(bbox_3d_2, axis=0)
    bbox_2_min = np.amin(bbox_3d_2, axis=0)

    overlap_min = np.maximum(bbox_1_min, bbox_2_min)
    overlap_max = np.minimum(bbox_1_max, bbox_2_max)

    # intersections and union
    if np.amin(overlap_max - overlap_min) < 0:
        intersections = 0
    else:
        intersections = np.prod(overlap_max - overlap_min)
    union = np.prod(bbox_1_max - bbox_1_min) + np.prod(bbox_2_max - bbox_2_min) - intersections
    overlaps = intersections / union
    return overlaps


def tensor_bbox_from_corners(corners, device):
    if not isinstance(corners, torch.Tensor):
        corners = torch.tensor(corners).to(device).float()
    pts = []
    for i in range(8):
        x, y, z = (i % 4) // 2, i // 4, i % 2
        cur_pt = torch.stack([corners[..., x, 0], corners[..., y, 1], corners[..., z, 2]], dim=-1)  # [Bs, 3]
        pts.append(cur_pt)
    return torch.stack(pts, dim=-2)  # [Bs, 8, 3]


def np_bbox_from_corners(corners):  # corners: [Bs, 2, 3]
    if isinstance(corners, np.ndarray):
        corners = torch.tensor(corners).float()
    bbox_shape = corners.shape[:-2] + (8, 3)  # [Bs, 8, 3]
    bbox = torch.zeros(*bbox_shape)
    for i in range(8):
        x, y, z = (i % 4) // 2, i // 4, i % 2
        bbox[..., i, 0] = corners[..., x, 0]
        bbox[..., i, 1] = corners[..., y, 1]
        bbox[..., i, 2] = corners[..., z, 2]
    return bbox


def yaxis_from_corners(corners, device):  # [B, P, 2, 3]
    if not isinstance(corners, torch.Tensor):
        corners = torch.tensor(corners).to(device).float()
    return corners * torch.tensor((0, 1, 0)).reshape(
        tuple(1 for _ in range(len(corners.shape) - 1)) + (3,)).float().to(corners.device)


def get_posed_bbox_from_part(part_model, corners):
    part_model = convert_part_model(part_model)
    bbox = tensor_bbox_from_corners(corners.detach(), corners.device) # [P, 2, 3] -> [P, 8, 3]?
    #  bbox = torch.tensor(bbox).to(part_model['scale'].device).float()

    posed_bbox = pose_with_part(part_model, bbox)
    if isinstance(posed_bbox, torch.Tensor):
        posed_bbox = posed_bbox.detach().cpu().numpy()

    return posed_bbox


def get_pred_nocs_corners(pred_seg, nocs_pred, num_parts):
    batch_size = len(pred_seg)
    nocs_pred = nocs_pred.detach().cpu().numpy()
    pred_seg = pred_seg.detach().cpu().numpy()
    all = []
    for b in range(batch_size):
        tmp = []
        for j in range(num_parts):
            idx = np.where(pred_seg[b] == j)[0]
            if len(idx) == 0:
                pred_corners = np.zeros((2, 3))
            else:
                centered_nocs = nocs_pred[b][idx]
                size_pred = np.max(abs(centered_nocs), axis=0)  # [3]
                pred_corners = np.stack([-size_pred, size_pred], axis=0)  # [2, 3]
            tmp.append(pred_corners)
        all.append(np.stack(tmp, axis=0))  # [P, 2, 3]
    all = np.stack(all, axis=0)
    return all


def calc_part_iou_list(gt_bbox_list, pred_bbox, separate='both', nocs=False):
    def to_numpy(x):
        if not isinstance(x, np.ndarray):
            return x.detach().cpu().numpy()
        else:
            return x
    iou_protocol = nocs_iou_3d if nocs else iou_3d
    gt_bbox_list = [to_numpy(gt_bbox) for gt_bbox in gt_bbox_list]
    pred_bbox = to_numpy(pred_bbox)
    batch_size, num_parts = pred_bbox.shape[:2]
    iou = {}
    per_iou = {}
    for part in range(num_parts):
        tmp = []
        for b in range(batch_size):
            max_iou = 0.0
            for gt_bbox in gt_bbox_list:
                cur_iou = iou_protocol(gt_bbox[b][part], pred_bbox[b][part])
                max_iou = max(max_iou, cur_iou)
            tmp.append(max_iou)
        tmp = np.array(tmp)
        per_iou[part] = tmp
        iou[part] = np.mean(tmp)

    if separate == True:
        return per_iou
    elif separate == False:
        return iou
    elif separate == 'both':
        return iou, per_iou


def eval_single_part_iou(gt_corners, pred_corners, gt_pose, pred_pose, separate=False, nocs=False, sym=False):
    gt_npcs_bbox = tensor_bbox_from_corners(gt_corners.detach(), gt_corners.device).cpu().numpy()
    pred_npcs_bbox = tensor_bbox_from_corners(pred_corners.detach(), gt_corners.device).cpu().numpy()

    if sym:
        def y_rotation_matrix(theta):
            np_mat = np.array([[np.cos(theta), 0, np.sin(theta)],
                               [0, 1, 0],
                               [-np.sin(theta), 0, np.cos(theta)]])
            return torch.tensor(np_mat).float().to(gt_corners.device)

        n = 20
        gt_poses = []  # pred_pose: 'rotation': [B, P, 3, 3]
        for i in range(n):
            cur_pose = {key: gt_pose[key].clone() for key in ['translation', 'scale']}
            cur_pose['rotation'] = torch.matmul(pred_pose['rotation'],
                                                y_rotation_matrix(2 * np.pi * i / float(n)).reshape(1, 1, 3, 3))
            gt_poses.append(cur_pose)
    else:
        gt_poses = [gt_pose]

    pred_posed_bbox = get_posed_bbox_from_part(pred_pose, pred_corners)
    pred_posed_gt_bbox = get_posed_bbox_from_part(pred_pose, gt_corners)
    gt_posed_bboxes = [get_posed_bbox_from_part(pose, gt_corners) for pose in gt_poses]

    npcs_iou = calc_part_iou_list([gt_npcs_bbox], pred_npcs_bbox, separate='both', nocs=nocs)
    iou = calc_part_iou_list(gt_posed_bboxes, pred_posed_bbox, separate='both', nocs=nocs)
    gt_bbox_iou = calc_part_iou_list(gt_posed_bboxes, pred_posed_gt_bbox, separate='both', nocs=nocs)

    ret_dict, per_ret_dict = {}, {}
    for iou_name, src in zip(['npcs_iou', 'iou', 'gt_bbox_iou'], [npcs_iou, iou, gt_bbox_iou]):
        ret_dict[iou_name], per_ret_dict[iou_name] = src

    if separate == True:
        return ret_dict
    elif separate == False:
        return per_ret_dict
    else:
        return ret_dict, per_ret_dict

