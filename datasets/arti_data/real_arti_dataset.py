import os
import sys
import numpy as np
from os.path import join as pjoin
import argparse
import json

base_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(base_dir, '..'))

from data_utils import farthest_point_sample, split_real_dataset
from utils import ensure_dirs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yml', help='path to config.yml')
    parser.add_argument('--root_dset',
                        default='../../sapien_data',
                        help='root_dset')
    parser.add_argument('--obj_category', default='laptop', help='name of the dataset we use')
    parser.add_argument('--experiment_dir', default='laptop', help='name of the dataset we use')
    parser.add_argument('--output_path', type=str, default=None)
    parser.add_argument('--num_points', type=int, default=4096)
    parser.add_argument('--show_rc', action='store_true')
    parser.add_argument('--random_camera', action='store_true')
    parser.add_argument('--is_debug', action='store_true')
    parser.add_argument('--range_pair', type=str, default=None)
    parser.add_argument('--mode', type=str, default='val')
    return parser.parse_args()


def generate_full_data(root_dset, obj_category, instance, track_num, frame_i, num_points):
    preproc_path = pjoin(root_dset, 'preproc',
                         obj_category, instance, f'{track_num}', 'full')
    ensure_dirs(preproc_path)
    cur_preproc_path = pjoin(preproc_path, f'{frame_i}.npz')
    if os.path.exists(cur_preproc_path):
        all = np.load(cur_preproc_path, allow_pickle=True)
        points = all['point']
        ret = {'points': points}
        if 'pose' in all:
            ret.update({'nocs2camera': [pose for pose in all['pose']], 'nocs_corners': all['corners']})

    cloud_path = pjoin(root_dset, 'render',
                 obj_category, instance, f'{track_num}', 'cloud', f'{frame_i}.npz')

    points = np.load(cloud_path, allow_pickle=True)['point']

    while len(points) < num_points:
        points = np.concatenate([points, points], axis=0)
    fps_idx = farthest_point_sample(points, num_points, device='cuda:0')
    points = points[fps_idx]

    ret = {'points': points}

    pose_path = pjoin(root_dset, 'real_pose', obj_category, instance, f'{track_num}.json')
    meta_path = pjoin(root_dset, 'real_pose', obj_category, instance, 'meta.json')
    if os.path.exists(pose_path) and os.path.exists(meta_path):
        with open(pose_path, 'r') as f:
            all_pose = json.load(f)
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        if obj_category == 'drawers':
            num_parts = 4
            name2num = {'drawer3': 0, 'drawer2': 1, 'drawer1': 2, 'body': 3}
        num2name = {value: key for key, value in name2num.items()}
        extents = np.stack([meta[num2name[p]]['size'] for p in range(num_parts)], axis=0)  # [P, 3]
        radius = np.sqrt(np.sum(extents ** 2, axis=-1))  # [P]
        extents /= radius.reshape(num_parts, 1)
        corners = np.stack([-extents * 0.5, extents * 0.5], axis=1)  # [P, 2, 3]

        mat = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
        nocs2camera = [{'rotation': np.matmul(mat, np.array(all_pose[int(frame_i)][num2name[p]]['R']).reshape(3, 3)),
                        'translation': np.matmul(mat, np.array(all_pose[int(frame_i)][num2name[p]]['t']).reshape(3, 1)),
                        'scale': float(radius[p])} for p in range(num_parts)]
        np.savez_compressed(cur_preproc_path, point=points, pose=nocs2camera, corners=corners)
        ret.update({'nocs2camera': nocs2camera, 'nocs_corners': corners})
    else:
        np.savez_compressed(cur_preproc_path, point=points)
    return ret


class SAPIENRealDataset:
    def __init__(self, root_dset, obj_category, obj_info, num_expr, num_points=4096,
                 truncate_length=None, is_debug=False):
        self.root_dset = root_dset
        self.obj_category = obj_category
        self.num_expr = num_expr
        self.obj_info = obj_info
        self.num_points = num_points
        self.file_list = self.collect_data(truncate_length)
        self.len = len(self.file_list)
        self.model_info_dict = {}
        self.ins_info = {}
        self.is_debug = is_debug

    def collect_data(self, truncate_length):
        splits_path = pjoin(self.root_dset, "splits", self.obj_category, self.num_expr)
        idx_txt = pjoin(splits_path, "real_test.txt")
        splits_ready = os.path.exists(idx_txt)
        if not splits_ready:
            split_real_dataset(self.root_dset, self.obj_category, self.num_expr, self.obj_info['real_test_list'])

        with open(idx_txt, "r", errors='replace') as fp:
            lines = fp.readlines()
            file_list = [line.strip() for line in lines]
        if truncate_length is not None:
            file_list = file_list[:truncate_length]

        return file_list

    def __getitem__(self, index):
        path = self.file_list[index]
        last_name_list = path.split('.')[-2].split('/')
        instance, track_num, _, frame_i = last_name_list[-4:]
        fake_last_name = '/'.join(last_name_list[:-2] + last_name_list[-1:])
        fake_path = f'{fake_last_name}.pkl'
        if instance not in self.ins_info:
            self.ins_info[instance] = None

        full_data = generate_full_data(self.root_dset, self.obj_category, instance, track_num, frame_i,
                                       self.num_points)
        meta = {'path': fake_path}
        if 'nocs2camera' in full_data:
            meta['nocs2camera'] = full_data.pop('nocs2camera')
        if 'nocs_corners' in full_data:
            meta['nocs_corners'] = full_data.pop('nocs_corners')
        return {'data': full_data,
                'meta': meta}

    def __len__(self):
        return self.len


