import os
import sys

import numpy as np
import cv2
import pickle
from os.path import join as pjoin
import argparse
from multiprocessing import Process

BASEPATH = os.path.dirname(__file__)
sys.path.insert(0, pjoin(BASEPATH, '..'))
sys.path.insert(0, pjoin(BASEPATH, '..', '..', '..'))

from utils import ensure_dirs
from nocs_utils import backproject, project, get_corners, bbox_from_corners
from tqdm import tqdm


def gather_instances(list_path, data_path, model_path, output_path, instances, intrinsics,
                     flip=True, real=False):
    for instance in tqdm(instances):
        gather_instance(list_path, data_path, model_path, output_path,
                        instance, intrinsics, flip=flip, real=real)


def gather_instance(list_path, data_path, model_path, output_path, instance, intrinsics,
                    flip=True, real=False, img_per_folder=100, render_rgb=False):
    corners = np.load(pjoin(model_path, f'{instance}.npy'))
    bbox = bbox_from_corners(corners)
    bbox *= 1.4
    meta_path = pjoin(list_path, f'{instance}.txt')
    with open(meta_path, 'r') as f:
        lines = f.readlines()

    inst_output_path = pjoin(output_path, instance)
    if not real:
        folder_num, img_num = 0, -1
        cur_folder_path = pjoin(inst_output_path, f'{folder_num:04d}')
        ensure_dirs([pjoin(cur_folder_path, name) for name in (['data'] if not render_rgb else ['rgb', 'data'])])

    meta_dict = {}

    for line in tqdm(lines, desc=f'Instance {instance}'):
        track_name, prefix = line.strip().split('/')[:2]
        file_path = pjoin(data_path, track_name)
        if real and track_name not in meta_dict:
            meta_dict[track_name] = file_path
        suffix = 'depth' if real else 'composed'
        try:
            depth = cv2.imread(pjoin(file_path, f'{prefix}_{suffix}.png'), -1)
            mask = cv2.imread(pjoin(file_path, f'{prefix}_mask.png'))[:, :, 2]
            rgb = cv2.imread(pjoin(file_path, f'{prefix}_color.png'))
            with open(pjoin(file_path, f'{prefix}_meta.txt'), 'r') as f:
                meta_lines = f.readlines()
            with open(pjoin(file_path, f'{prefix}_pose.pkl'), 'rb') as f:
                pose_dict = pickle.load(f)
        except:
            continue
        if flip:
            depth, mask, rgb = depth[:, ::-1], mask[:, ::-1], rgb[:, ::-1]
        inst_num = -1
        for meta_line in meta_lines:
            inst_num = int(meta_line.split()[0])
            inst_id = meta_line.split()[-1]
            if inst_id == instance:
                break
        if inst_num not in pose_dict:
            continue
        pose = pose_dict[inst_num]
        posed_bbox = (np.matmul(bbox, pose['rotation'].swapaxes(-1, -2))
                      * np.expand_dims(pose['scale'], (-1, -2))
                      + pose['translation'].swapaxes(-1, -2))
        center = posed_bbox.mean(axis=0)
        radius = np.sqrt(np.sum((posed_bbox[0] - center) ** 2)) + 0.1
        aa_corner = get_corners([center - np.ones(3) * radius, center + np.ones(3) * radius])
        aabb = bbox_from_corners(aa_corner)
        height, width = mask.shape
        projected_corners = project(aabb, intrinsics).astype(np.int32)[:, [1, 0]]
        projected_corners[:, 0] = height - projected_corners[:, 0]
        corner_2d = np.stack([np.min(projected_corners, axis=0),
                              np.max(projected_corners, axis=0)], axis=0)
        corner_2d[0, :] = np.maximum(corner_2d[0, :], 0)
        corner_2d[1, :] = np.minimum(corner_2d[1, :], np.array([height - 1, width - 1]))
        corner_mask = np.zeros_like(mask)
        corner_mask[corner_2d[0, 0]: corner_2d[1, 0] + 1, corner_2d[0, 1]: corner_2d[1, 1] + 1] = 1
        cropped_rgb = rgb[corner_2d[0, 0]: corner_2d[1, 0] + 1, corner_2d[0, 1]: corner_2d[1, 1] + 1]

        raw_pts, raw_idx = backproject(depth, intrinsics=intrinsics, mask=corner_mask)
        raw_mask = (mask == inst_num)[raw_idx[0], raw_idx[1]]

        def filter_box(pts, corner):
            mask = np.prod(np.concatenate([pts >= corner[0], pts <= corner[1]], axis=1).astype(np.int8),  # [N, 6]
                           axis=1)
            idx = np.where(mask == 1)[0]
            return pts[idx]

        def filter_ball(pts, center, radius):
            distance = np.sqrt(np.sum((pts - center) ** 2, axis=-1))  # [N]
            idx = np.where(distance <= radius)
            return pts[idx], idx

        pts, idx = filter_ball(raw_pts, center, radius)
        obj_mask = raw_mask[idx]

        data_dict = {'points': pts, 'labels': obj_mask, 'pose': pose,
                     'path': pjoin(file_path, f'{prefix}_{suffix}.png')}
        if not real:
            img_num += 1
            if img_num >= img_per_folder:
                folder_num += 1
                cur_folder_path = pjoin(inst_output_path, f'{folder_num:04d}')
                ensure_dirs([pjoin(cur_folder_path, name) for name in (['data'] if not render_rgb else ['rgb', 'data'])])
                img_num = 0
            np.savez_compressed(pjoin(cur_folder_path, 'data', f'{img_num:02d}.npz'), all_dict=data_dict)
            if render_rgb:
                cv2.imwrite(pjoin(cur_folder_path, 'rgb', f'{img_num:02d}.png'), cropped_rgb)
        else:
            cur_folder_path = pjoin(inst_output_path, track_name)
            ensure_dirs(pjoin(cur_folder_path, 'data'))
            np.savez_compressed(pjoin(cur_folder_path, 'data', f'{prefix}.npz'), all_dict=data_dict)

    if real:
        cur_folder_path = pjoin(inst_output_path, track_name)
        ensure_dirs([cur_folder_path])
        for track_name in meta_dict:
            with open(pjoin(cur_folder_path, 'meta.txt'), 'w') as f:
                print(meta_dict[track_name], file=f)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../../nocs_data')
    parser.add_argument('--data_type', type=str, default='train')
    parser.add_argument('--list_path', type=str, default='../../nocs_data/instance_list')
    parser.add_argument('--model_path', type=str, default='../../nocs_data/model_corners')
    parser.add_argument('--output_path', type=str, default='../../nocs_data/instance_data')
    parser.add_argument('--category', type=int, default=1)
    parser.add_argument('--parallel', action='store_true')
    parser.add_argument('--num_proc', type=int, default=10)

    return parser.parse_args()


def main(args):
    if args.data_type in ['real_train', 'real_test']:
        intrinsics = np.array([[591.0125, 0, 322.525], [0, 590.16775, 244.11084], [0, 0, 1]])
    else:
        intrinsics = np.array([[577.5, 0, 319.5], [0., 577.5, 239.5], [0., 0., 1.]])

    data_path = pjoin(args.data_path, args.data_type)
    list_path = pjoin(args.list_path, args.data_type, str(args.category))
    model_path = args.model_path
    output_path = pjoin(args.output_path, args.data_type, str(args.category))
    ensure_dirs(output_path)
    instances = list(map(lambda s: s.split('.')[0], os.listdir(list_path)))
    instances.sort()

    if not args.parallel:
        gather_instances(list_path, data_path, model_path, output_path, instances, intrinsics,
                         flip=args.data_type not in ['real_train', 'real_test'],
                         real=args.data_type in ['real_train', 'real_test'])
    else:
        processes = []
        proc_cnt = args.num_proc
        num_per_proc = int((len(instances) - 1) / proc_cnt) + 1

        for k in range(proc_cnt):
            s_ind = num_per_proc * k
            e_ind = min(num_per_proc * (k + 1), len(instances))
            p = Process(target=gather_instances,
                        args=(list_path, data_path, model_path, output_path,
                              instances[s_ind: e_ind], intrinsics,
                              args.data_type not in ['real_train', 'real_test'],
                              args.data_type in ['real_train', 'real_test']))
            processes.append(p)
            p.start()

        """
        for process in processes:
            process.join()
        """


if __name__ == '__main__':
    args = parse_args()
    main(args)

