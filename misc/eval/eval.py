import sys
import os
from os.path import join as pjoin

sys.path.insert(0, pjoin(os.path.dirname(__file__), '..'))
sys.path.insert(0, pjoin(os.path.dirname(__file__), '..', '..'))
import numpy as np
import argparse
import pickle

from configs.config import get_config
from pose_utils.metrics import rot_diff_degree
from pose_utils.part_dof_utils import eval_part_full
from pose_utils.bbox_utils import eval_single_part_iou
from utils import cvt_torch, log_loss_summary, per_dict_to_csv, add_dict


def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--config', type=str, default='config.yml', help='path to config.yml')
    parser.add_argument('--obj_config', type=str, default=None, help='path to obj_config.yml')
    parser.add_argument('--obj_category', type=str, default=None, help='path to obj_config.yml')
    parser.add_argument('--experiment_dir', type=str, default=None, help='root dir for all outputs')
    return parser.parse_args()


def eval_data(name, data, obj_info):
    poses, corners = cvt_torch(data['pred']['poses'], 'cpu'), cvt_torch(data['pred']['corners'], 'cpu')
    gt_poses, gt_corners = cvt_torch(data['gt']['poses'], 'cpu'), cvt_torch(data['gt']['corners'], 'cpu')

    error_dict = {}
    sym = obj_info['sym']
    rigid = obj_info['num_parts'] == 1

    for i in range(len(poses)):
        if i == 0:  # the first frame's pose is given by initialization
            continue
        key = f'{name}_{i}'
        _, per_diff = eval_part_full(gt_poses[i], poses[i], per_instance=True, yaxis_only=sym)
        error_dict[key] = {key: float(value.numpy()) for key, value in per_diff.items()}
        _, per_iou = eval_single_part_iou(gt_corners.unsqueeze(0), corners[i].unsqueeze(0),
                                          {key: value.unsqueeze(0) for key, value in gt_poses[i].items()},
                                          {key: value.unsqueeze(0) for key, value in poses[i].items()},
                                          separate='both',
                                          nocs=rigid, sym=sym)
        per_iou = {f'iou_{j}': float(per_iou['iou'][j]) for j in range(len(per_iou['iou']))}
        error_dict[key].update(per_iou)

        if not rigid:
            joint_state = get_joint_state(obj_info, poses[i])
            gt_joint_state = get_joint_state(obj_info, gt_poses[i])

            joint_diff = np.abs(joint_state - gt_joint_state)
            error_dict[key].update({f'theta_diff_{j}': joint_diff[j] for j in range(len(joint_diff))})

    return error_dict


def get_joint_state(info, pred_pose):
    tree = info['tree']
    joint_states = []
    for c, p in enumerate(tree):
        if p == -1:
            continue
        if info['type'] == 'revolute':
            state = rot_diff_degree(pred_pose['rotation'][c],
                                    pred_pose['rotation'][p])
        else:
            p_rot = pred_pose['rotation'][p]
            p_trans = pred_pose['translation'][p]
            c_trans = pred_pose['translation'][c]
            relative_trans = np.matmul(p_rot.transpose(-1, -2), c_trans - p_trans)
            axis_index = info['main_axis'][len(joint_states)]
            axis = np.zeros((3, ))
            axis[axis_index] = 1
            state = np.dot(relative_trans.reshape(-1), axis)
        joint_states.append(state)
    return np.array(joint_states)


if __name__ == "__main__":
    args = parse_args()
    cfg = get_config(args, save=False)
    base_path = cfg['obj']['basepath']
    obj_category = cfg['obj_category']

    obj_info = cfg['obj_info']

    data_path = pjoin(cfg['experiment_dir'], 'results', 'data')

    all_raw = os.listdir(data_path)
    all_raw = sorted(all_raw)

    error_dict = {}

    for i, raw in enumerate(all_raw):
        name = raw.split('.')[-2]
        with open(pjoin(data_path, raw), 'rb') as f:
            data = pickle.load(f)
        cur_dict = eval_data(name, data, obj_info)
        error_dict.update(cur_dict)

    err_path = pjoin(cfg['experiment_dir'], 'results', 'err.pkl')
    with open(err_path, 'wb') as f:
        pickle.dump(error_dict, f)
    avg_dict = {}
    for inst in error_dict:
        add_dict(avg_dict, error_dict[inst])
    log_loss_summary(avg_dict, len(error_dict), lambda x, y: print(f'{x}: {y}'))
    per_dict_to_csv(error_dict, err_path.replace('pkl', 'csv'))

