import os
import sys

import numpy as np
import cv2
import pickle
from os.path import join as pjoin
import argparse
from multiprocessing import Process
from tqdm import tqdm

BASEPATH = os.path.dirname(__file__)
sys.path.insert(0, BASEPATH)
sys.path.insert(0, pjoin(BASEPATH, '..'))

from nocs_utils import backproject, remove_border
from align_pose import pose_fit


def get_image_pose(num_instances, mask, coord, depth, intrinsics):
    pose_dict = {}
    for i in range(1, num_instances + 1):
        if np.sum(mask == i) < 3:
            continue
        pts, idxs = backproject(depth, intrinsics, mask == i)
        coord_pts = coord[idxs[0], idxs[1], :]  # already centered
        if len(pts) < 3:
            continue
        # plot3d_pts([[pts], [coord_pts]])
        pose = pose_fit(coord_pts, pts)
        if pose is not None:
            pose_dict[i] = pose

    return pose_dict


def get_pose(root_path, folders, intrinsics, flip=True, real=False):
    for sub_folder in tqdm(folders):
        file_path = pjoin(root_path, sub_folder)
        if not os.path.isdir(file_path):
            continue
        valid_data = [file[:4] for file in os.listdir(file_path) if file.endswith('color.png')]
        valid_data.sort()
        for prefix in valid_data:
            """
            if real or not os.path.exists(pjoin(file_path, f'{prefix}_composed.png')):
                depth = cv2.imread(pjoin(file_path, f'{prefix}_depth.png'), -1)
                if len(depth.shape) == 3:
                    depth = np.uint16(depth[:, :, 1] * 256) + np.uint16(depth[:, :, 2])
                    depth = depth.astype(np.uint16)
            else:
                depth = cv2.imread(pjoin(file_path, f'{prefix}_composed.png'), -1)
            """
            depth = cv2.imread(pjoin(file_path, f'{prefix}_depth.png'), -1)
            coord = cv2.imread(pjoin(file_path, f'{prefix}_coord.png'))
            mask = cv2.imread(pjoin(file_path, f'{prefix}_mask.png'))
            if depth is None or coord is None or mask is None:
                print(pjoin(file_path, f'{prefix}_depth.png'))
                continue
            if len(depth.shape) == 3:
                depth = np.uint16(depth[:, :, 1] * 256) + np.uint16(depth[:, :, 2])
                depth = depth.astype(np.uint16)
            mask = mask[:, :, 2]

            if flip:
                depth, coord, mask = depth[:, ::-1], coord[:, ::-1], mask[:, ::-1]

            if real:
                mask = remove_border(mask, kernel_size=2)

            # plot_images([coord, mask])

            coord = coord[:, :, (2, 1, 0)]
            coord = coord / 255. - 0.5
            if not flip:
                coord[..., 2] = -coord[..., 2]   # verify!!!

            with open(pjoin(file_path, f'{prefix}_meta.txt'), 'r') as f:
                lines = f.readlines()
            poses = get_image_pose(len(lines), mask, coord, depth, intrinsics)
            with open(pjoin(file_path, f'{prefix}_pose.pkl'), 'wb') as f:
                pickle.dump(poses, f)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../../nocs_data')
    parser.add_argument('--data_type', type=str, default='train')
    parser.add_argument('--parallel', action='store_true')
    parser.add_argument('--num_proc', type=int, default=10)

    return parser.parse_args()


def main(args):
    root_path = pjoin(args.data_path, args.data_type)
    folders = os.listdir(root_path)
    folders.sort()

    if args.data_type in ['real_train', 'real_test']:
        intrinsics = np.array([[591.0125, 0, 322.525], [0, 590.16775, 244.11084], [0, 0, 1]])
    else:
        intrinsics = np.array([[577.5, 0, 319.5], [0., 577.5, 239.5], [0., 0., 1.]])

    if not args.parallel:
        get_pose(root_path, folders, intrinsics, flip=args.data_type in ['train', 'val'],
                 real=args.data_type in ['real_train', 'real_test'])
    else:
        processes = []
        proc_cnt = args.num_proc
        num_per_proc = int((len(folders) - 1) / proc_cnt) + 1

        for k in range(proc_cnt):
            s_ind = num_per_proc * k
            e_ind = min(num_per_proc * (k + 1), len(folders))
            p = Process(target=get_pose,
                        args=(root_path, folders[s_ind: e_ind], intrinsics, args.data_type in ['train', 'val']))
            processes.append(p)
            p.start()

        """
        for process in processes:
            process.join()
        """


if __name__ == '__main__':
    args = parse_args()
    main(args)

