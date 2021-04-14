import os
import sys
base_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(base_dir, '..'))
sys.path.append(os.path.join(base_dir, '..', '..'))
import numpy as np
import argparse
import cv2
import pickle
from os.path import join as pjoin
from utils import ensure_dirs

"""
render/laptop/real_0/track_num/
    rgb
    cloud

"""

def process_pkl(pkl_path, output_path):
    for folder in ['rgb', 'cloud']:
        ensure_dirs(pjoin(output_path, folder))
    with open(pkl_path, 'rb') as f:
        all_dict = pickle.load(f)  # 'point_cloud', 'image', 'time'
    points_list, image_list = all_dict['point_cloud'], all_dict['image']
    num_frames = len(points_list)

    for i in range(num_frames):
        img = image_list[i]
        raw_point = points_list[i]
        cv2.imwrite(pjoin(output_path, 'rgb', f'{i}.png'), img)
        point = np.stack([raw_point[..., 2], -raw_point[..., 0], -raw_point[..., 1]], axis=-1)
        # [[0, 0, 1], [-1, 0, 0], [0, -1, 0]]
        # plot3d_pts([[point]], show_fig=True)
        np.savez_compressed(pjoin(output_path, 'cloud', f'{i}.npz'), point=point)


def batch_process_laptop():
    input_path = '../../robotic_data/output_data'
    instances = ['kinect2', 'realsense']
    num_tracks = 8
    output_path = '../../sapien_data/render/laptop'

    for track_num in range(num_tracks):
        for instance in instances:
            cur_pkl_path = pjoin(input_path, f'{track_num}_{instance}.pkl')
            cur_output_path = pjoin(output_path, f'{instance}_0', f'{track_num}')
            process_pkl(cur_pkl_path, cur_output_path)


def batch_process_drawers():
    input_path = '../../robotic_data/drawers/data'
    instances = ['kinect2', 'realsense']
    track_nums = [14, 15]
    output_path = '../../sapien_data_robotic/render/drawers'

    for track_num in track_nums:
        for instance in instances:
            cur_pkl_path = pjoin(input_path, f'{track_num}_{instance}.pkl')
            cur_output_path = pjoin(output_path, f'{instance}_0', f'{track_num}')
            process_pkl(cur_pkl_path, cur_output_path)


if __name__ == '__main__':
    batch_process_drawers()
