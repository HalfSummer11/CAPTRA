import sys
import os
from os.path import join as pjoin
BASEPATH = os.path.dirname(__file__)
sys.path.insert(0, BASEPATH)
sys.path.insert(0, pjoin(BASEPATH, '..'))
sys.path.insert(0, pjoin(BASEPATH, '..', '..'))
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from backbones import PointNet2Msg
from blocks import get_point_mlp, RotationRegressor
from pose_utils.part_dof_utils import merge_reenact_canon_part_pose, convert_pred_rtvec_to_matrix
from pose_utils.procrustes import scale_pts_mask, translate_pts_mask, transform_pts_2d_mask, rot_around_yaxis_to_3d
from pose_utils.pose_fit import part_fit_st_no_ransac


class CoordNet(nn.Module):
    def __init__(self, cfg):
        super(CoordNet, self).__init__()
        self.backbone = PointNet2Msg(cfg, cfg['network']['backbone_out_dim'],
                                     net_type='camera', use_xyz_feat=True)
        in_dim = cfg['network']['backbone_out_dim']
        num_parts = cfg['num_parts']
        self.num_parts = num_parts
        self.sym = cfg['obj_sym']
        seg_dim = self.num_parts + cfg['obj']['extra_dims']
        self.seg_head = get_point_mlp(in_dim, seg_dim, [], acti='none', dropout=None)
        self.nocs_head = get_point_mlp(in_dim, 3 * num_parts,
                                       cfg['network']['nocs_head_dims'],
                                       acti='sigmoid', dropout=None)

    def forward(self, input, test=False):
        cam = input['points']  # [B, 3, N]
        points_mean = input['points_mean']  # [B, 3, 1]
        canon_pose = input['canon_pose']
        cam = cam + points_mean
        cam = cam - canon_pose['translation']
        cam = torch.matmul(canon_pose['rotation'].transpose(-1, -2), cam)  # [B, 3, 3] [B, 3, N]
        cam = cam / canon_pose['scale'].unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1]

        feat = self.backbone(cam)  # [B, backbone_out_dim, N]
        seg = self.seg_head(feat)  # [B, P, N]
        seg = F.softmax(seg, dim=1)
        nocs = self.nocs_head(feat) - 0.5

        pred_dict = {
            'seg': seg,
            'nocs': nocs,
            'points': cam
        }

        if 'gt_part' in input:  # compute s, r, t!
            pred_labels = torch.argmax(seg, dim=-2)
            labels = pred_labels if test else input['labels']
            rotation = input['gt_part']['rotation']
            final_pose = {'rotation': rotation}
            pred_npcs = nocs.reshape(len(nocs), self.num_parts, 3, -1)
            cam_points = input['points'] + input['points_mean']  # [B, 3, N]
            cam_points = cam_points.unsqueeze(1).repeat(1, self.num_parts, 1, 1)

            eye = torch.cat([torch.eye(self.num_parts), torch.zeros(2, self.num_parts)], dim=0).to(pred_npcs.device)
            mask = eye[labels, ].transpose(-1, -2)  # [B, N, P] --> [B, P, N]
            valid_mask = (mask.sum(dim=-1) > 0).float()

            init_part = input['init_part']

            if self.sym:
                canon_cam = torch.matmul(rotation.transpose(-1, -2), cam_points)   # [B, P, 3, N]
                src_2d = pred_npcs[..., [0, 2], :].transpose(-1, -2)  # [B, P, N, 2]
                tgt_2d = canon_cam[..., [0, 2], :].transpose(-1, -2)
                rot_2d, _ = transform_pts_2d_mask(src_2d, tgt_2d, mask.unsqueeze(-1))
                rot_3d = rot_around_yaxis_to_3d(rot_2d)
                rotated_npcs = torch.matmul(rotation, torch.matmul(rot_3d, pred_npcs))
            else:
                rotated_npcs = torch.matmul(rotation, pred_npcs)

            scale_mask = mask.unsqueeze(-2)  # [B, P, 1, N]

            def center(source, mask):
                source_center = torch.sum(source * mask, dim=-1, keepdim=True) / torch.clamp(torch.sum(mask, dim=-1,
                                                                                                       keepdim=True),
                                                                                             min=1.0)
                source_centered = (source - source_center.detach()) * mask  # [B, P, 3, N]
                return source_centered

            final_pose['scale'] = scale_pts_mask(center(rotated_npcs, scale_mask),
                                                 center(cam_points, scale_mask),
                                                 scale_mask)
            final_pose['scale'] = (valid_mask * final_pose['scale']
                                   + (1.0 - valid_mask) * init_part['scale'])
            invalid_scale_mask = torch.logical_or(torch.isnan(final_pose['scale']),
                                                  torch.isinf(final_pose['scale'])).float()
            final_pose['scale'] = (1.0 - invalid_scale_mask) * final_pose['scale'] + invalid_scale_mask * init_part['scale']

            scale = final_pose['scale'] if test else input['gt_part']['scale']

            scaled_npcs = scale.unsqueeze(-1).unsqueeze(-1) * rotated_npcs  # [B, P, 3, N]
            final_pose['translation'] = translate_pts_mask(scaled_npcs, cam_points, mask.unsqueeze(-1))

            final_pose['translation'] = (valid_mask.unsqueeze(-1).unsqueeze(-1) * final_pose['translation']
                                         + (1.0 - valid_mask.unsqueeze(-1).unsqueeze(-1)) * init_part['translation'])
            invalid_trans_mask = torch.logical_or(torch.isnan(final_pose['translation'].sum((-1, -2))),
                                                  torch.isinf(final_pose['translation'].sum((-1, -2)))).float().unsqueeze(-1).unsqueeze(-1)
            final_pose['translation'] = (1.0 - invalid_trans_mask) * final_pose['translation'] + invalid_trans_mask * init_part['translation']

            pred_dict['part'] = final_pose

        return pred_dict


class RotationRegressionBackbone(nn.Module):
    def __init__(self, cfg):
        super(RotationRegressionBackbone, self).__init__()
        self.num_parts = cfg['num_parts']
        self.encoder = PointNet2Msg(cfg, cfg['network']['backbone_out_dim'], use_xyz_feat=False)
        self.sym = cfg['obj_sym']
        self.pose_pred = RotationRegressor(cfg['network']['backbone_out_dim'], self.num_parts,
                                           symmetric=self.sym)
        self.cfg = cfg

    def forward(self, cam, cam_labels):  # [B, 3, N], [B, N]
        feat = self.encoder(cam)
        labels = cam_labels

        eye_mat = torch.cat([torch.eye(self.num_parts), torch.zeros(2, self.num_parts)], dim=0)
        part_mask = eye_mat[labels, ].to(labels.device).transpose(-1, -2).unsqueeze(-2)  # [B, P, 1, N]
        valid_mask = (part_mask.sum(dim=(-1, -2)) > 0).float().unsqueeze(-1)  # [B, P, 1]

        raw_pred = self.pose_pred(feat)  # [B, x, N]
        weighted_pred = (raw_pred * part_mask).sum(-1) / torch.clamp_min(part_mask.sum(-1), 1.0)  # [B, P, D] / [B, P, 1]
        if self.sym:
            default = torch.tensor((0, 1, 0))
        else:
            default = torch.eye(3).reshape(-1)
        weighted_pred = (valid_mask * weighted_pred
                         + (1.0 - valid_mask) * default.float().to(raw_pred.device).reshape(1, 1, -1))
        pred = {'rtvec': weighted_pred, 'point_rtvec': raw_pred}

        return pred


class PartCanonNet(nn.Module):
    def __init__(self, cfg):
        super(PartCanonNet, self).__init__()
        self.type = cfg['network']['type']
        self.regress_net = RotationRegressionBackbone(cfg)
        self.device = cfg['device']
        self.num_parts = cfg['num_parts']
        self.sym = cfg['obj_sym']
        self.tree = cfg['obj_tree']
        self.root = [i for i in range(self.num_parts) if self.tree[i] == -1][0]
        self.cfg = cfg

    def forward(self, input, test_mode=False):
        """
        If "eval_baseline rnpcs", pass pred_labels and pred_nocs by input
        """
        state = input['state']
        part_pose = state['part']

        if 'canon_pose' in input:
            canon_pose = input['canon_pose']
        else:
            part_pose = input['state']['part']
            canon_pose = {key: part_pose[key].reshape((-1,) + part_pose[key].shape[2:])
                          for key in ['rotation', 'translation', 'scale']}

        cam = input['points']  # [B, 3, N] or [B, 3, P * N] -- if cam_point_type == per_part
        points_mean = input['points_mean']  # [B, 3, 1]
        eval_rnpcs = self.type == 'rot_coord_track'
        if eval_rnpcs:
            cam_seg = input['pred_labels']
        else:
            cam_seg = input['labels']
        batch_size = len(cam)

        cam = cam.unsqueeze(1).repeat(1, self.num_parts, 1, 1).reshape((-1, ) + cam.shape[-2:])  # [B, 1 -> P, 3, N]
        cam_seg = cam_seg.unsqueeze(1).repeat(1, self.num_parts, 1).reshape((-1, ) + cam_seg.shape[-1:])  # [B, 1-> P, N] --> [B * P, N]
        points_mean = points_mean.unsqueeze(1).repeat(1, self.num_parts, 1, 1).reshape(
            (-1, ) + points_mean.shape[-2:])

        cam = cam + points_mean
        cam = cam - canon_pose['translation']
        cam = torch.matmul(canon_pose['rotation'].transpose(-1, -2), cam)   # [B, 3, 3] [B, 3, N]
        cam = cam / canon_pose['scale'].unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1]

        pred = self.regress_net(cam, cam_seg)

        rtvec = pred.pop('rtvec')
        pred['rotation'] = convert_pred_rtvec_to_matrix(rtvec, self.sym)

        rtvec = pred.pop('point_rtvec')  # either [B, D, N] or [B, P, D, N]
        if len(rtvec.shape) == 3:
            rtvec = rtvec.unsqueeze(1)
        rtvec = rtvec.transpose(-1, -2)  # [B, P/1, N, D]
        pred['point_rotation'] = convert_pred_rtvec_to_matrix(rtvec, self.sym)

        for key in pred.keys():
            raw = pred[key].reshape((batch_size, self.num_parts) + pred[key].shape[1:])  # [B, P, P, x]
            idx = torch.tensor(range(self.num_parts)).long().to(self.device)
            pred[key] = raw[:, idx, idx]

        if self.type == 'rot':
            final_pose = merge_reenact_canon_part_pose(part_pose, pred)
            for key in ['translation', 'scale']:
                final_pose[key] = input['gt_part'][key].detach().clone()

        if eval_rnpcs:
            merged_pose = merge_reenact_canon_part_pose(part_pose, pred)
            rotation = merged_pose['rotation']

            final_pose = {'rotation': rotation}

            pred_labels = input['pred_labels']
            pred_npcs = input['pred_nocs'].reshape(batch_size, self.num_parts, 3, -1)

            cam_points = input['points'] + input['points_mean']  # [B, 3, N]
            cam_points = cam_points.unsqueeze(1).repeat(1, self.num_parts, 1, 1)

            labels = pred_labels if test_mode else input['labels']
            rotation = final_pose['rotation'] if test_mode else input['gt_part']['rotation']

            scale = None

            final_pose, valid = part_fit_st_no_ransac(labels, pred_npcs.transpose(-1, -2), cam_points.transpose(-1, -2),
                                               rotation, {'num_parts': self.num_parts, 'sym': self.sym}, given_scale=scale)

            final_pose['scale'] = valid.float() * final_pose['scale'] + (1.0 - valid.float()) * part_pose['scale']
            final_pose['translation'] = (valid.float().unsqueeze(-1).unsqueeze(-1) * final_pose['translation']
                                         + (1.0 - valid.float().unsqueeze(-1).unsqueeze(-1)) * part_pose['translation'])

        ret_dict = {'part': final_pose}

        for key in ['point_trans', 'point_rotation', 'point_scale', 'seg', 'nocs', 'avg_nocs', 'pred_labels']:
            if key in pred:
                ret_dict[key] = pred[key]

        return ret_dict


def parse_args():
    parser = argparse.ArgumentParser('SingleFrameModel')
    parser.add_argument('--config', type=str, default='config.yml', help='path to config.yml')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

