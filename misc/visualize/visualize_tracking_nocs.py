import sys
import os
from os.path import join as pjoin

sys.path.insert(0, pjoin(os.path.dirname(__file__), '..'))
sys.path.insert(0, pjoin(os.path.dirname(__file__), '..', '..'))
import numpy as np
import torch
import argparse
import pickle
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt

from misc.visualize.vis_utils import plot_bboxes_on_image
from datasets.nocs_data.nocs_utils import project
from pose_utils.part_dof_utils import reenact_with_part
from pose_utils.bbox_utils import np_bbox_from_corners

category_list = ['bottle', 'bowl', 'camera', 'can', 'laptop', 'mug']
exp_list = [f'{i + 1}_{category}_rot' for i, category in enumerate(category_list)]

def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--img_path', type=str, default='../data/nocs_data/nocs_full/real_test')
    parser.add_argument('--exp_path', type=str, default='../runs')
    parser.add_argument('--output_path', type=str, default='../nocs_viz')
    parser.add_argument('--scene', type=str, default=None)
    parser.add_argument('--show_fig', action='store_true', default=False, help='show figure')
    parser.add_argument('--save_fig', action='store_true', default=False, help='save figure')
    parser.add_argument('--depth', action='store_true', default=False, help='save figure')
    return parser.parse_args()


def plot_sequence(img_path, data_dict, output_path, scene_name, cam_intrinsics, scale=1000.0,
                  show_fig=False, save_fig=True, depth=False):

    all_ins = list(data_dict.keys())
    frame_nums = []
    for ins in all_ins:
        frame_nums += [frame_num[0] for frame_num in data_dict[ins]['frame_nums']]
    frame_nums = list(np.unique(frame_nums))
    frame_nums.sort()

    frame_dict = {}

    for ins in all_ins:
        frame_dict[ins] = {data_dict[ins]['frame_nums'][i][0]: i for i in range(len(data_dict[ins]['frame_nums']))}

    def get_posed_bbox(pose, bbox):
        pose = {key: torch.tensor(value).float() for key, value in pose.items()}
        return reenact_with_part({'points': bbox}, pose)

    image_suffix = 'depth' if depth else 'color'
    for j, frame_num in tqdm(enumerate(frame_nums)):
        image_path = pjoin(img_path, scene_name, f'{frame_num}_{image_suffix}.png')
        if depth:
            image = cv2.imread(image_path, -1)
            image = np.stack([image, image, image], axis=-1)
        else:
            image = cv2.imread(image_path)[..., ::-1]
        out_path = pjoin(output_path, scene_name, f'{frame_num}.png')
        bbox_list = []
        for ins in all_ins:
            cur_data = data_dict[ins]
            if frame_num not in frame_dict[ins]:
                continue
            i = frame_dict[ins][frame_num]
            corners = cur_data['gt']['corners'] if i == 0 else cur_data['pred']['corners'][i]
            bbox = np_bbox_from_corners(corners)
            pose = cur_data['pred']['poses'][i]
            posed_pred_bbox = get_posed_bbox(pose, bbox)
            bboxes = posed_pred_bbox.reshape(-1, 3)
            if isinstance(bboxes, torch.Tensor):
                bboxes = bboxes.cpu().numpy()
            img_bbox = project(bboxes, cam_intrinsics, scale)
            h = len(image)
            img_bbox[..., 1] = h - img_bbox[..., 1]
            img_bbox = img_bbox.reshape(8, 2)
            bbox_list.append(img_bbox)

        bbox_list = np.stack(bbox_list, axis=0)
        plot_bboxes_on_image(image, bbox_list, show_fig=show_fig, save_fig=save_fig, out_path=out_path)


def plot_all_sequences(img_path, exp_list, output_path, scene_list, cam_intrinsics, scale=1000.0,
                       show_fig=False, save_fig=False, depth=False):
    scene_dict = {scene_name: {} for scene_name in scene_list}
    for exp_dir in exp_list:
        result_path = pjoin(exp_dir, 'results', 'data')
        sequences = [seq for seq in os.listdir(result_path) if seq.endswith('.pkl')]
        for seq in sequences:
            tmp_seq = seq.split('.')[-2].split('_')
            instance, scene_name = '_'.join(tmp_seq[:3]), '_'.join(tmp_seq[-2:])
            if scene_name in scene_dict:
                with open(pjoin(result_path, seq), 'rb') as f:
                    scene_dict[scene_name][instance] = pickle.load(f)
    for scene_name in scene_list:
        plot_sequence(img_path, scene_dict[scene_name], output_path, scene_name, cam_intrinsics, scale=scale,
                      show_fig=show_fig, save_fig=save_fig, depth=depth)


if __name__ == "__main__":
    args = parse_args()

    intrinsics = np.array([[591.0125, 0, 322.525], [0, 590.16775, 244.11084], [0, 0, 1]])
    if args.scene is None:
        scene_list = [f'scene_{i}' for i in range(1, 7)]
    else:
        scene_list = [args.scene]

    plot_all_sequences(args.img_path, [pjoin(args.exp_path, exp_name) for exp_name in exp_list], args.output_path, scene_list,
                       intrinsics, scale=1000.0, show_fig=False, save_fig=True, depth=args.depth)

