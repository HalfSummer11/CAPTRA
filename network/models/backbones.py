import os
import sys
from os.path import join as pjoin
BASEPATH = os.path.dirname(__file__)
sys.path.insert(0, pjoin(BASEPATH, '..'))
sys.path.insert(0, pjoin(BASEPATH, '..', '..'))
import argparse
import torch.nn as nn
import torch
import torch.nn.functional as F

from pointnet_utils import PointNetSetAbstraction, PointNetFeaturePropagation, PointNetSetAbstractionMsg


class PointNet2Msg(nn.Module):
    """
    Output: 128 channels
    """
    def __init__(self, cfg, out_dim, net_type='camera', use_xyz_feat=False):
        super(PointNet2Msg, self).__init__()
        net_cfg = cfg['pointnet'][net_type]
        self.out_dim = out_dim
        self.in_dim = 3 if use_xyz_feat else 0
        self.use_xyz_feat = use_xyz_feat
        self.sa1 = PointNetSetAbstractionMsg(npoint=net_cfg['sa1']['npoint'],
                                             radius_list=net_cfg['sa1']['radius_list'],
                                             nsample_list=net_cfg['sa1']['nsample_list'],
                                             in_channel=self.in_dim + 3,
                                             mlp_list=net_cfg['sa1']['mlp_list'])

        self.sa2 = PointNetSetAbstractionMsg(npoint=net_cfg['sa2']['npoint'],
                                             radius_list=net_cfg['sa2']['radius_list'],
                                             nsample_list=net_cfg['sa2']['nsample_list'],
                                             in_channel=self.sa1.out_channel + 3,
                                             mlp_list=net_cfg['sa2']['mlp_list'])

        self.sa3 = PointNetSetAbstraction(npoint=None,
                                          radius=None,
                                          nsample=None,
                                          in_channel=self.sa2.out_channel + 3,
                                          mlp=net_cfg['sa3']['mlp'], group_all=True)

        self.fp3 = PointNetFeaturePropagation(in_channel=(self.sa2.out_channel + self.sa3.out_channel),
                                              mlp=net_cfg['fp3']['mlp'])
        self.fp2 = PointNetFeaturePropagation(in_channel=(self.sa1.out_channel + self.fp3.out_channel),
                                              mlp=net_cfg['fp2']['mlp'])
        self.fp1 = PointNetFeaturePropagation(in_channel=(self.in_dim + 3 + self.fp2.out_channel),
                                              mlp=net_cfg['fp1']['mlp'])

        self.conv1 = nn.Conv1d(self.fp1.out_channel, self.out_dim, 1)
        self.bn1 = nn.BatchNorm1d(self.out_dim)

        self.device = cfg['device']

    def forward(self, input):  # [B, 3, N]
        l0_xyz = input
        if self.use_xyz_feat:
            l0_points = input
        else:
            l0_points = input[:, 3:]
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, torch.cat([l0_xyz, l0_points], dim=1), l1_points)
        feat = F.relu(self.bn1(self.conv1(l0_points)))
        return feat


def parse_args():
    parser = argparse.ArgumentParser('SingleFrameModel')
    parser.add_argument('--config', type=str, default='config.yml', help='path to config.yml')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    from configs.config import get_config
    cfg = get_config(args, save=False)
    msg = PointNet2Msg(cfg, out_dim=128)
    print(msg)


