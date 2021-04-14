import os
import sys

import numpy as np
import cv2
from os.path import join as pjoin
import argparse
from tqdm import tqdm

BASEPATH = os.path.dirname(__file__)
sys.path.insert(0, pjoin(BASEPATH, '..', '..', '..'))

from utils import ensure_dirs


def get_valid_instance(root_path, folders, real, min_points=50):
    data_list = {cls_id: {} for cls_id in range(1, 7)}
    for sub_folder in tqdm(folders):
        file_path = pjoin(root_path, sub_folder)
        if not os.path.isdir(file_path):
            continue
        valid_data = [file[:4] for file in os.listdir(file_path) if file.endswith('color.png')]
        valid_data.sort()
        for prefix in valid_data:
            # filter instances: should belong to class > 0, should appear in mask w/ at least 50 pixels
            mask_path = pjoin(file_path, f'{prefix}_mask.png')
            meta_path = pjoin(file_path, f'{prefix}_meta.txt')
            if not os.path.exists(mask_path) or not os.path.exists(meta_path):
                # print(mask_path, 'does not exist')
                continue
            mask = cv2.imread(mask_path)[:, :, 2]
            # appeared_ins = list(np.unique(mask))
            with open(pjoin(meta_path), 'r') as f:
                lines = f.readlines()
            for line in lines:
                if real:
                    inst_num, cls_id, inst_id = line.split()[:3]
                    inst_id = inst_id.split('.')[0].replace('/', '_')
                else:
                    inst_num, cls_id, cls_code, inst_id = line.split()[:4]
                inst_num, cls_id = int(inst_num), int(cls_id)
                cnt = np.sum(mask == inst_num)
                if cls_id == 0 or cnt < min_points:
                    continue
                if inst_id not in data_list[cls_id]:
                    data_list[cls_id][inst_id] = []
                data_list[cls_id][inst_id].append(f'{sub_folder}/{prefix}')

    return data_list


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../../nocs_data')
    parser.add_argument('--data_type', type=str, default='train')
    parser.add_argument('--list_path', type=str, default='../../nocs_data/instance_list')
    parser.add_argument('--parallel', action='store_true')
    parser.add_argument('--num_proc', type=int, default=10)

    return parser.parse_args()


def main(args):
    root_path = pjoin(args.data_path, args.data_type)
    folders = os.listdir(root_path)  # [folder for folder in os.listdir(root_path) if os.path.isdir(pjoin(root_path, folder))]
    folders.sort()
    output_path = pjoin(args.list_path, args.data_type)
    ensure_dirs(output_path)

    data_list = get_valid_instance(root_path, folders, real=args.data_type in ['real_train', 'real_test'])

    for cls_id in data_list:
        cur_path = pjoin(output_path, str(cls_id))
        ensure_dirs(cur_path)
        for instance_id in data_list[cls_id]:
            with open(pjoin(cur_path, f'{instance_id}.txt'), 'w') as f:
                for line in data_list[cls_id][instance_id]:
                    print(line, file=f)


if __name__ == '__main__':
    args = parse_args()
    main(args)

