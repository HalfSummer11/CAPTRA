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
import matplotlib.pyplot as plt
from tqdm import tqdm

from misc.visualize.vis_utils import plot_bbox_with_pts, compute_lims_for_pts
from datasets.data_utils import get_pts_for_bbox_overlay
from configs.config import get_config
from pose_utils.part_dof_utils import reenact_with_part
from pose_utils.bbox_utils import np_bbox_from_corners

"""
exp_dir/
    results/
        data/
            instance_seqnum.pkl
        viz/
"""


def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--config', type=str, default='config_track.yml')
    parser.add_argument('--obj_config', type=str, default='obj_info_sapien.yml')
    parser.add_argument('--obj_category', type=str, default=None, help='laptop')
    parser.add_argument('--experiment_dir', type=str, default=None, help='../runs/laptop_rot')
    parser.add_argument('--show_fig', action='store_true', default=False, help='show figure')
    parser.add_argument('--save_fig', action='store_true', default=False, help='save figure')
    return parser.parse_args()


def plot_sequence(cfg, base_path, obj_category, instance, seq_num, input_path, output_path,
                  show_fig=False, save_fig=True):
    with open(input_path, 'rb') as f:
        data = pickle.load(f)
    poses = data['pred']['poses']

    gt_bbox = np_bbox_from_corners(data['gt']['corners'])
    pred_bbox = [gt_bbox] + [np_bbox_from_corners(corners) for corners in data['pred']['corners'][1:]]

    bbox = pred_bbox

    def get_posed_bbox(pose, bbox):
        pose = {key: torch.tensor(value) for key, value in pose.items()}
        posed_pred_bbox = reenact_with_part({'points': bbox}, pose)
        return posed_pred_bbox

    pts_list = []
    for i in range(len(poses)):
        pts = get_pts_for_bbox_overlay(base_path, obj_category, instance, seq_num, str(i),
                                       num_parts=cfg['num_parts'], num_points=cfg['num_points'],
                                       synthetic=cfg['obj']['synthetic'], suffix='_seq')
        pts_list.append(pts)
    lims = compute_lims_for_pts(np.concatenate(pts_list))

    for i in tqdm(range(len(poses))):
        posed_pred_bbox = get_posed_bbox(poses[i], bbox[i])
        pts = pts_list[i]
        bboxes = posed_pred_bbox.reshape(-1, 8, 3)
        if isinstance(bboxes, torch.Tensor):
            bboxes = bboxes.cpu().numpy()

        out_path = pjoin(output_path, f'{instance}_{seq_num}_{i:03}.png')
        plot_bbox_with_pts(pts, bboxes, lims=lims,
                           show_fig=show_fig, save_fig=save_fig, save_path=out_path)


def plot_all_sequences(cfg, base_path, obj_category, exp_dir, show_fig=False, save_fig=True):
    data_path = pjoin(exp_dir, 'results', 'data')
    seq_names = [name for name in os.listdir(data_path) if name.endswith('.pkl')]
    for seq_name in seq_names:
        input_path = pjoin(exp_dir, 'results', 'data', seq_name)
        output_path = pjoin(exp_dir, 'results', 'viz', 'image', seq_name.split('.')[-2])
        instance, seq_num = seq_name.split('.')[-2].split('_')
        plot_sequence(cfg, base_path, obj_category, instance, seq_num, input_path, output_path,
                      show_fig, save_fig)


if __name__ == "__main__":
    args = parse_args()
    cfg = get_config(args, save=False)
    base_path = cfg['obj']['basepath']
    exp_dir = os.path.abspath(args.experiment_dir)

    plot_all_sequences(cfg, base_path, cfg['obj_category'], exp_dir,
                       show_fig=args.show_fig, save_fig=args.save_fig)

