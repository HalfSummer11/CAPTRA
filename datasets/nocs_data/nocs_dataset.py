import os
import sys
import numpy as np
from os.path import join as pjoin
import argparse
import cv2

base_dir = os.path.dirname(__file__)
sys.path.append(base_dir)
sys.path.append(os.path.join(base_dir, '..'))
sys.path.append(os.path.join(base_dir, '..', '..'))

from nocs_data_process import base_generate_data, split_nocs_dataset, read_cloud
from misc.visualize.vis_utils import plot3d_pts
from tqdm import tqdm
from configs.config import get_config

"""
NOCS data organization:
render/
    train|val/
        1|2|3|4|5|6/
            instance/
                0000~0999/
                    data/
                        00.pkl (sometimes start with 01.pkl)
                        01.pkl
                    rgb/
    real_test/
        1|2|3|4|5|6/
            instance/
                scene_1-scene_6
                    data/
                        0xxx.pkl
                    meta.txt
"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yml', help='path to config.yml')
    parser.add_argument('--obj_config', type=str, default='config.yml', help='path to config.yml')
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


def generate_nocs_corners(root_dset, instance):
    obj_path = pjoin(root_dset, 'model_corners', f'{instance}.npy')
    nocs_corners = np.load(obj_path).reshape(1, -1, 3)  # [1, 2, 3]
    return nocs_corners


def generate_nocs_data(root_dset, mode, obj_category, instance, track_num, frame_i, num_points, radius, perturb_cfg,
                       device):

    path = pjoin(root_dset, 'render', mode, obj_category, instance, f'{track_num}')
    cloud_path = pjoin(path, 'data', f'{frame_i}.npz')
    cloud_dict = np.load(cloud_path, allow_pickle=True)['all_dict'].item()

    cam_points, seg, perturbed_pose = read_cloud(cloud_dict, num_points, radius, perturb_cfg, device)
    if cam_points is None:
        return None
    full_data = base_generate_data(cam_points, seg, cloud_dict['pose'])
    full_data['ori_path'] = cloud_dict['path']
    full_data['crop_pose'] = [perturbed_pose]
    if 'real' in mode:
        depth_path = cloud_dict['path']
        depth = cv2.imread(depth_path, -1)
        with open(depth_path.replace('depth.png', 'meta.txt'), 'r') as f:
            meta_lines = f.readlines()
        inst_num = -1
        for meta_line in meta_lines:
            inst_num = int(meta_line.split()[0])
            inst_id = meta_line.split()[-1]
            if inst_id == instance:
                break
        mask = cv2.imread(depth_path.replace('depth', 'mask'))[:, :, 2]
        mask = (mask == inst_num)
        full_data['pre_fetched'] = {'depth': depth.astype(np.int16), 'mask': mask}
    else:
        full_data['pre_fetched'] = {}

    return full_data


def read_nocs_pose(root_dset, mode, obj_category, instance, track_num, frame_i):

    path = pjoin(root_dset, 'render', mode, obj_category, instance, f'{track_num}')
    cloud_path = pjoin(path, 'data', f'{frame_i}.npz')
    cloud_dict = np.load(cloud_path, allow_pickle=True)['all_dict'].item()
    pose = cloud_dict['pose']
    return pose


class NOCSDataset:
    def __init__(self, root_dset, obj_category, obj_info, num_expr, num_points=4096, mode='train',
                 truncate_length=None, synthetic=True, radius=0.6, perturb_cfg=None, device=None, downsampling=None):
        self.root_dset = root_dset
        self.obj_category = obj_category
        self.num_expr = num_expr
        self.obj_info = obj_info
        self.num_points = num_points
        self.mode = mode
        self.bad_ins = obj_info['bad_ins']
        self.file_list = self.collect_data(truncate_length, downsampling)
        self.len = len(self.file_list)
        self.nocs_corner_dict = {}
        self.ins_info = {}
        self.synthetic = synthetic
        self.radius = radius
        self.perturb_cfg = perturb_cfg
        self.invalid_dict = {}
        self.device = device

    def collect_data(self, truncate_length, downsampling):
        splits_path = pjoin(self.root_dset, "splits", self.obj_category, self.num_expr)
        idx_txt = pjoin(splits_path, f'{self.mode}.txt')
        splits_ready = os.path.exists(idx_txt)
        if not splits_ready:
            split_nocs_dataset(self.root_dset, self.obj_category, self.num_expr, self.mode, self.bad_ins)
        with open(idx_txt, "r", errors='replace') as fp:
            lines = fp.readlines()
            file_list = [line.strip() for line in lines]

        if downsampling is not None:
            file_list = file_list[::downsampling]

        if truncate_length is not None:
            file_list = file_list[:truncate_length]

        return file_list

    def get_pose(self, index):
        path = self.file_list[index]
        last_name_list = path.split('.')[-2].split('/')
        instance, track_num, _, frame_i = last_name_list[-4:]
        return read_nocs_pose(self.root_dset, self.mode, self.obj_category, instance, track_num, frame_i)

    def __getitem__(self, index):
        path = self.file_list[index]
        last_name_list = path.split('.')[-2].split('/')
        instance, track_num, _, frame_i = last_name_list[-4:]
        fake_last_name = '/'.join(last_name_list[:-2] + last_name_list[-1:])
        fake_path = f'{fake_last_name}.pkl'
        if instance not in self.nocs_corner_dict:
            self.nocs_corner_dict[instance] = generate_nocs_corners(self.root_dset, instance)
        if index not in self.invalid_dict:
            full_data = generate_nocs_data(self.root_dset, self.mode, self.obj_category, instance,
                                           track_num, frame_i, self.num_points, self.radius,
                                           self.perturb_cfg, self.device)
            if full_data is None:
                self.invalid_dict[index] = True
        if index in self.invalid_dict:
            return self.__getitem__((index + 1) % self.len)

        nocs2camera = full_data.pop('nocs2camera')
        crop_pose = full_data.pop('crop_pose')
        ori_path = full_data.pop('ori_path')
        pre_fetched = full_data.pop('pre_fetched')
        return {'data': full_data,
                'meta': {'path': fake_path,
                         'ori_path': ori_path,
                         'nocs2camera': nocs2camera,
                         'crop_pose': crop_pose,
                         'pre_fetched': pre_fetched,
                         'nocs_corners': self.nocs_corner_dict[instance]}
               }

    def __len__(self):
        return self.len

    def save_all_items(self, range_pair=None):
        range_i = range(len(self)) if range_pair is None else range(*range_pair)
        for i in tqdm(range_i):
            self.__getitem__(i)


def visualize_data(data_dict):
    points, labels, nocs = data_dict['data']['points'], data_dict['data']['labels'], data_dict['data']['nocs']
    nocs2camera = data_dict['meta']['nocs2camera']

    def pose_points(pose_dict, pts, labels):
        num_parts = len(pose_dict)
        res = np.zeros_like(pts)
        for p in range(num_parts):
            idx = np.where(labels == p)[0]
            cur = pts[idx].transpose(-1, -2)
            pose = pose_dict[p]
            cur = pose['scale'] * np.matmul(pose['rotation'], cur) + pose['translation']
            res[idx] = cur.transpose(-1, -2)
        return res

    def group_pts(pts, labels):
        max_l = np.max(labels)
        pt_list = []
        for p in range(max_l + 1):
            idx = np.where(labels == p)
            pt_list.append(pts[idx])
        return pt_list

    posed_nocs = pose_points(nocs2camera, nocs, labels)

    path = data_dict['meta']['path']
    category, instance, track, frame_num = path.split('.')[-2].split('/')[-4:]
    plot3d_pts([group_pts(points, labels), group_pts(posed_nocs, labels), group_pts(nocs, labels)],
               show_fig=True, save_fig=True,
               save_path=pjoin('nocs_data_vis', category),
               save_title='_'.join([instance, track, frame_num]))


def main(args):
    cfg = get_config(args, save=False)

    dataset = NOCSDataset(cfg['obj']['basepath'],
                          cfg['obj_category'],
                          cfg['obj_info'], cfg['num_expr'], mode=args.mode)
    ins_dict = {}
    lim = 5
    for i in range(len(dataset)):
        path = dataset.file_list[i]
        instance = path.split('.')[-2].split('/')[-3]
        if instance not in ins_dict:
            ins_dict[instance] = 0
        ins_dict[instance] += 1
        if ins_dict[instance] <= lim:
            print(path)
            visualize_data(dataset[i])
    for i in tqdm(range(len(dataset))):
        cur_pose = dataset.get_pose(i)
        path = dataset.file_list[i]
        for key, value in cur_pose.items():
            if np.any(np.isnan(value)) or np.any(np.isinf(value)):
                print('inf or nan in', key)
                print(path)
        if np.any(cur_pose['scale'] < 1e-3):
            print('scale too small!', cur_pose['scale'])
            print(path)


if __name__ == '__main__':
    args = parse_args()
    main(args)
