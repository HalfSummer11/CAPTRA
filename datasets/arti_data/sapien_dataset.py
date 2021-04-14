import os
import sys
import numpy as np
from os.path import join as pjoin
import pickle
import argparse

base_dir = os.path.dirname(__file__)
sys.path.append(base_dir)
sys.path.append(os.path.join(base_dir, '..'))

from data_utils import read_gt_pose_dict, split_dataset, split_seq_dataset
from arti_data_process import base_generate_data, read_cloud, generate_instance_info
from misc.visualize.vis_utils import plot3d_pts
from tqdm import tqdm
from configs.config import get_config


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


def generate_full_data(root_dset, obj_category, instance, track_num, frame_i, model_info, num_parts, num_points,
                       save_full=True, is_debug=False, synthetic=False, suffix='', perturb=False, device=None):
    num_parts = None if not synthetic else num_parts
    preproc_name = f'preproc{suffix}'
    preproc_path = pjoin(root_dset, preproc_name,
                         obj_category, instance, f'{track_num}')
    full_preproc_path = pjoin(preproc_path, 'full')
    cloud_preproc_path = pjoin(preproc_path, 'cloud')
    for path in [full_preproc_path, cloud_preproc_path]:
        if not os.path.exists(path):
            os.makedirs(path)

    cur_full_preproc_path = pjoin(full_preproc_path, f'{frame_i}.pkl')
    if os.path.exists(cur_full_preproc_path):
        with open(cur_full_preproc_path, 'rb') as f:
            full_data = pickle.load(f)
        return full_data

    render_name = f'render{suffix}'
    path = pjoin(root_dset, render_name,
                 obj_category, instance, f'{track_num}')
    cloud_path = pjoin(path, 'cloud', f'{frame_i}.npz')
    gt_path = pjoin(path, 'gt', f'{frame_i}.pkl')

    cur_cloud_preproc_path = pjoin(cloud_preproc_path, f'{frame_i}.pkl')
    if os.path.exists(cur_cloud_preproc_path):
        with open(cur_cloud_preproc_path, 'rb') as f:
            preproc_dict = pickle.load(f)
            cam_points, seg = preproc_dict['cam'], preproc_dict['seg']
    else:
        cloud_dict = np.load(cloud_path, allow_pickle=True)['all_dict'].item()
        cam_points, seg = read_cloud(cloud_dict, num_points, synthetic=synthetic, num_parts=num_parts,
                                     perturb=perturb, device=device)
        with open(cur_cloud_preproc_path, 'wb') as f:
            pickle.dump({'cam': cam_points, 'seg': seg}, f)

    with open(gt_path, 'rb') as f:
        gt_dict = pickle.load(f)
        cam2world, link2world_dict = read_gt_pose_dict(gt_dict)

    full_data = base_generate_data(model_info, cam_points, seg, cam2world, link2world_dict, is_debug=is_debug)

    if save_full:
        with open(cur_full_preproc_path, 'wb') as f:
            pickle.dump(full_data, f)

    return full_data


class SAPIENDataset:
    def __init__(self, root_dset, obj_category, obj_info, num_expr, num_points=4096, mode='train',
                 truncate_length=None, is_debug=False, synthetic=False, perturb=False, device=None):
        self.root_dset = root_dset
        self.obj_category = obj_category
        self.num_expr = num_expr
        self.obj_info = obj_info
        self.num_parts = obj_info['num_parts']
        self.num_points = num_points
        self.mode = mode
        self.syn_seq = mode in ['train_seq', 'test_seq']
        self.suffix = '_seq' if self.syn_seq else ''
        self.file_list = self.collect_data(truncate_length)
        self.len = len(self.file_list)
        self.model_info_dict = {}
        self.ins_info = {}
        self.is_debug = is_debug
        self.synthetic = synthetic
        self.perturb = perturb
        self.device = device

    def collect_data(self, truncate_length):
        splits_path = pjoin(self.root_dset, "splits", self.obj_category, self.num_expr)

        if not self.syn_seq:
            splits_ready = os.path.exists(pjoin(splits_path, "val.txt"))
            if not splits_ready:
                split_dataset(self.root_dset, self.obj_category, self.num_expr,
                              test_ins=self.obj_info['test_list'], temporal=True)
        else:
            if not os.path.exists(pjoin(splits_path, "test_seq.txt")):
                split_seq_dataset(self.root_dset, self.obj_category, self.num_expr,
                                  test_ins=self.obj_info['test_list'])

        idx_txt = pjoin(splits_path, f'{self.mode}.txt')
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
        if instance not in self.model_info_dict:
            self.model_info_dict[instance] = generate_instance_info(self.root_dset, self.obj_category, instance)

        model_info = self.model_info_dict[instance]
        if instance not in self.ins_info:
            self.ins_info[instance] = {
                'corners': [model_info['global_corner']] + model_info['corner']}

        full_data = generate_full_data(self.root_dset, self.obj_category, instance, track_num, frame_i,
                                       model_info, self.num_parts, self.num_points,
                                       is_debug=self.is_debug,
                                       synthetic=self.synthetic, suffix=self.suffix,
                                       perturb=self.perturb, device=self.device)

        nocs2camera = full_data.pop('nocs2camera')
        return {'data': full_data,
                'meta': {'path': fake_path,
                         'nocs2camera': nocs2camera}
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

    plot3d_pts([[points], group_pts(points, labels), group_pts(posed_nocs, labels), group_pts(nocs, labels)])


def main(args):
    cfg = get_config(args, save=False)

    bla = SAPIENDataset(args.root_dset, args.obj_category,
                        cfg['obj_info'], cfg['num_expr'], mode=args.mode,
                        is_debug=args.is_debug)
    for j in range(10):
        i = np.random.randint(0, len(bla))
        print(bla.file_list[i])
        visualize_data(bla[i])


if __name__ == '__main__':
    args = parse_args()
    main(args)
