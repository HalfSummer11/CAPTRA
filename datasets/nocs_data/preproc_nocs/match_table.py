import os
import sys

import numpy as np
import cv2
from os.path import join as pjoin
import argparse
from multiprocessing import Process
from tqdm import tqdm


class TableDict:
    def __init__(self, table_path='../../nocs_data/ikea_data'):
        self.path = table_path
        self.load_all_tables()
        self.table_dict, self.table_mat, self.idx_list = self.load_all_tables()

    def load_all_tables(self):
        table_path = self.path
        folders = [folder for folder in os.listdir(table_path) if os.path.isdir(os.path.join(table_path, folder))]
        table_dict = {}
        color_table, source = [], []
        for folder in folders:
            table_num = int(folder[5:])
            table_dict[table_num] = {}
            cur_dict = table_dict[table_num]
            for img in os.listdir(pjoin(table_path, folder)):
                cur_path = pjoin(table_path, folder, img)
                if not img.endswith('png'):
                    continue
                img_num = int(img[:4])
                if img_num not in cur_dict:
                    cur_dict[img_num] = {}
                if img.endswith('depth.png'):
                    cur_dict[img_num]['depth'] = cv2.imread(cur_path, -1)  # [480, 640], uint16
                elif img.endswith('color.png'):
                    cur_dict[img_num]['color'] = cv2.imread(cur_path)
                    color_table.append(cur_dict[img_num]['color'])
                    source.append(f'{table_num}_{img_num}')

        color_table = np.stack(color_table, axis=0)
        return table_dict, color_table, source

    def look_up(self, img, mask):
        residual = (self.table_mat - np.expand_dims(img, 0)) * np.expand_dims(mask, (0, -1))
        residual = residual.sum(axis=(1, 2, 3))
        idx = np.argmin(residual)
        source = self.idx_list[idx]
        table_num, img_num = map(int, source.split('_'))
        rgb = self.table_dict[table_num][img_num]['color']
        depth = self.table_dict[table_num][img_num]['depth']
        return rgb, depth


def compose_depth(root_path, folders, table):
    for sub_folder in tqdm(folders):
        file_path = pjoin(root_path, sub_folder)
        valid_data = [file[:4] for file in os.listdir(file_path) if file.endswith('color.png')]
        valid_data.sort()
        for prefix in valid_data:
            rgb = cv2.imread(pjoin(file_path, f'{prefix}_color.png'))
            depth = cv2.imread(pjoin(file_path, f'{prefix}_depth.png'))
            if len(depth.shape) == 3:
                depth_16 = np.uint16(depth[:, :, 1] * 256) + np.uint16(depth[:, :, 2])
                depth_16 = depth_16.astype(np.uint16)
            elif len(depth.shape) == 2 and depth.dtype == 'uint16':
                depth_16 = depth

            raw_mask = cv2.imread(pjoin(file_path, f'{prefix}_mask.png'))[:, :, 2]
            bi_mask = (raw_mask == 255).astype(np.uint16)
            bg_rgb, bg_depth = table.look_up(rgb, bi_mask)
            composed_depth = bi_mask * bg_depth + (1 - bi_mask) * depth_16

            cv2.imwrite(pjoin(file_path, f'{prefix}_composed.png'), composed_depth)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../../nocs_data/train')
    parser.add_argument('--bg_path', type=str, default='../../nocs_data/ikea_data')
    parser.add_argument('--parallel', action='store_true')
    parser.add_argument('--num_proc', type=int, default=10)

    return parser.parse_args()


def main(args):
    table = TableDict(args.bg_path)
    root_path = args.data_path
    folders = [folder for folder in os.listdir(root_path) if os.path.isdir(pjoin(root_path, folder))]
    folders.sort()

    if not args.parallel:
        compose_depth(root_path, folders, table)
    else:
        processes = []
        proc_cnt = args.num_proc
        num_per_proc = int((len(folders) - 1) / proc_cnt) + 1

        for k in range(proc_cnt):
            s_ind = num_per_proc * k
            e_ind = min(num_per_proc * (k + 1), len(folders))
            p = Process(target=compose_depth,
                        args=(root_path, folders[s_ind: e_ind], table))
            processes.append(p)
            p.start()

        """
        for process in processes:
            process.join()
        """


if __name__ == '__main__':
    args = parse_args()
    main(args)

