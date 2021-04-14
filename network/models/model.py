import torch.nn as nn
import torch
import pickle
import time
import os
import sys
import numpy as np
from copy import deepcopy
from os.path import join as pjoin
from abc import abstractmethod

BASEPATH = os.path.dirname(__file__)
sys.path.insert(0, BASEPATH)
sys.path.insert(0, pjoin(BASEPATH, '..'))
sys.path.insert(0, pjoin(BASEPATH, '..', '..'))
from networks import CoordNet, PartCanonNet
from loss import compute_miou_loss, compute_nocs_loss, choose_coord_by_label, \
    compute_part_dof_loss, rot_trace_loss, compute_point_pose_loss, rot_yaxis_loss
from pose_utils.bbox_utils import get_pred_nocs_corners, eval_single_part_iou, tensor_bbox_from_corners, yaxis_from_corners
from pose_utils.part_dof_utils import part_model_batch_to_part, eval_part_full, add_noise_to_part_dof, \
    compute_parts_delta_pose
from datasets.nocs_data.nocs_data_process import full_data_from_depth_image
from utils import Timer, add_dict, divide_dict, log_loss_summary, get_ith_from_batch, \
    cvt_torch, ensure_dirs, per_dict_to_csv


class BaseModel(nn.Module):
    def __init__(self, cfg):
        super(BaseModel, self).__init__()
        self.Net = None
        self.num_parts = int(cfg['num_parts'])
        self.num_joints = int(cfg['num_joints'])
        self.device = cfg['device']
        self.loss_weights = cfg['loss_weight']
        self.network_type = cfg['network']['type']
        raw_cfg = cfg['pose_perturb']
        self.pose_perturb_cfg = {'type': raw_cfg['type'],
                                 'scale': raw_cfg['s'],
                                 'translation': raw_cfg['t'],  # Now pass the sigma of the norm
                                 'rotation': np.deg2rad(raw_cfg['r'])}
        self.sym = cfg['obj_sym']
        self.cfg = cfg
        self.feed_dict = {}
        self.pred_dict = {}
        self.save_dict = {}
        self.loss_dict = {}
        self.per_diff_dict = {}

    def prepare_poses(self, data):
        gt_part = part_model_batch_to_part(cvt_torch(data['meta']['nocs2camera'], self.device), self.num_parts,
                                           self.device)
        init_part = add_noise_to_part_dof(gt_part, self.pose_perturb_cfg)
        if 'crop_pose' in data['meta']:
            crop_pose = part_model_batch_to_part(cvt_torch(data['meta']['crop_pose'], self.device),
                                                 self.num_parts, self.device)
            for key in ['translation', 'scale']:
                init_part[key] = crop_pose[key]
        return gt_part, init_part

    def summarize_losses(self, loss_dict):
        total_loss = 0
        for key, item in self.loss_weights.items():
            if key in loss_dict:
                total_loss += loss_dict[key] * item
        loss_dict['total_loss'] = total_loss
        self.loss_dict = loss_dict

    def record_per_diff(self, data, per_diff):
        paths = data['meta']['path']
        for i, path in enumerate(paths):
            instance, track_num, frame_i = path.split('.')[-2].split('/')[-3:]
            key = f'{instance}_{track_num}_{frame_i}'
            if key not in self.per_diff_dict:
                self.per_diff_dict[key] = {}
            self.per_diff_dict[key].update(get_ith_from_batch(per_diff, i))

    def save_per_diff(self):
        output_path = self.cfg['experiment_dir']
        timestamp = time.strftime("%m-%d-%H-%M-%S")
        with open(pjoin(output_path, f'{timestamp}.pkl'), 'wb') as f:
            pickle.dump(self.per_diff_dict, f)

        avg_dict = {}
        for inst in self.per_diff_dict:
            add_dict(avg_dict, self.per_diff_dict[inst])
        log_loss_summary(avg_dict, len(self.per_diff_dict), lambda x, y: print('Test_Real_Avg {} is {}'.format(x, y)))
        per_dict_to_csv(self.per_diff_dict, pjoin(output_path, f'{timestamp}.csv'))
        print(timestamp)

    @abstractmethod
    def set_data(self, data):
        pass

    @abstractmethod
    def compute_loss(self):
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def test(self, save=False, no_eval=False, epoch=0):
        pass


class CanonCoordModel(BaseModel):
    def __init__(self, cfg):
        super(CanonCoordModel, self).__init__(cfg)
        self.net = CoordNet(cfg)
        self.tree = cfg['obj_tree']
        self.root = [p for p in range(len(self.tree)) if self.tree[p] == -1][0]
        self.pwm_num = None if not self.sym else cfg['network']['pwm_num']
        self.pose_loss_type = cfg['pose_loss_type']
        self.cfg = cfg

    def set_data(self, data):
        self.feed_dict = {}
        for key, item in data.items():
            if key not in ['meta', 'labels', 'points', 'nocs']:
                continue
            elif key in ['meta']:
                pass
            elif key in ['labels']:
                item = item.long().to(self.device)
            else:
                item = item.float().to(self.device)
            self.feed_dict[key] = item
        self.feed_dict['points_mean'] = data['meta']['points_mean'].float().to(self.device)

    def prepare_data(self):
        data = self.feed_dict

        gt_part, init_part = self.prepare_poses(data)

        canon_pose = {key: init_part[key][:, self.root] for key in ['rotation', 'translation', 'scale']}

        self.feed_dict['canon_pose'] = canon_pose
        self.feed_dict['init_part'] = init_part
        self.feed_dict['gt_part'] = gt_part

    def compute_loss(self, test=False):
        feed_dict = self.feed_dict
        pred_dict = self.pred_dict
        loss_dict = {}

        seg_loss = compute_miou_loss(pred_dict['seg'], feed_dict['labels'],
                                     per_instance=False)
        loss_dict['seg_loss'] = seg_loss

        gt_labels = feed_dict['labels']
        pred_labels = torch.max(pred_dict['seg'], dim=-2)[1]  # [B, P, N] -> [B, N]
        labels = pred_labels if test else gt_labels

        nocs_loss = compute_nocs_loss(pred_dict['nocs'], feed_dict['nocs'],
                                      labels=labels,
                                      confidence=None, loss='l2', self_supervise=False,
                                      per_instance=False, sym=self.sym, pwm_num=self.pwm_num)
        if self.sym:
            loss_dict['nocs_dist_loss'], loss_dict['nocs_pwm_loss'] = nocs_loss
        else:
            loss_dict['nocs_loss'] = nocs_loss

        gt_corners = feed_dict['meta']['nocs_corners'].float().to(self.device)
        if self.sym:
            gt_bbox = yaxis_from_corners(gt_corners, self.device)
        else:
            gt_bbox = tensor_bbox_from_corners(gt_corners, self.device)

        pose_diff, per_diff = eval_part_full(feed_dict['gt_part'], pred_dict['part'],
                                             yaxis_only=self.sym)
        init_part_pose = feed_dict['init_part']
        init_pose_diff, init_per_diff = eval_part_full(feed_dict['gt_part'], init_part_pose,
                                                       yaxis_only=self.sym)
        loss_dict.update(pose_diff)
        loss_dict.update({f'init_{key}': value for key, value in init_pose_diff.items()})

        """loss"""

        loss_dict.update(compute_part_dof_loss(feed_dict['gt_part'], pred_dict['part'],
                                               self.pose_loss_type))

        corner_loss, corner_per_diff = compute_point_pose_loss(feed_dict['gt_part'], pred_dict['part'],
                                                               gt_bbox,
                                                               metric=self.pose_loss_type['point'])
        loss_dict['corner_loss'] = corner_loss

        self.summarize_losses(loss_dict)

    def test(self, save=False, no_eval=False, epoch=0):
        self.prepare_data()
        self.loss_dict = {}
        with torch.no_grad():
            self.pred_dict = self.net(self.feed_dict, test=True)
            if not no_eval:
                self.compute_loss(test=True)

    def update(self):
        self.prepare_data()
        self.pred_dict = self.net(self.feed_dict)
        self.compute_loss()
        self.loss_dict['total_loss'].backward()


class RotationModel(BaseModel):
    def __init__(self, cfg):
        super(RotationModel, self).__init__(cfg)
        self.net = PartCanonNet(cfg)
        self.pwm_num = None if not self.sym else cfg['network']['pwm_num']
        self.pose_loss_type = cfg['pose_loss_type']
        self.cfg = cfg
        self.raw_feed_dict = {}

    def set_data(self, data):
        self.raw_feed_dict = data

    def prepare_data(self, data):
        gt_part, init_part = self.prepare_poses(data)

        input = {'points': data['points'] ,
                 'points_mean': data['meta']['points_mean'],
                 'nocs': data['nocs'],
                 'state': {'part': init_part}, 'gt_part': gt_part}

        input = cvt_torch(input, self.device)
        input['meta'] = data['meta']
        input['labels'] = data['labels'].long().to(self.device)

        part_pose = input['state']['part']
        canon_pose = {key: part_pose[key].reshape((-1, ) + part_pose[key].shape[2:])  # [B, P, x] --> [B * P, x]
                      for key in ['rotation', 'translation', 'scale']}

        input['canon_pose'] = canon_pose

        batch_size = len(input['gt_part']['scale'])
        part_delta = compute_parts_delta_pose(input['state']['part'],
                                              input['gt_part'],
                                              {key: value.reshape((batch_size, self.num_parts) + value.shape[1:])
                                               for key, value in canon_pose.items()})
        input['root_delta'] = part_delta
        self.feed_dict = input

    def compute_loss(self, test_mode=False, per_instance=False):
        feed_dict = self.feed_dict
        pred_dict = self.pred_dict
        loss_dict = {}

        labels = self.feed_dict['labels']

        gt_corners = feed_dict['meta']['nocs_corners'].float().to(self.device)
        if self.sym:
            gt_bbox = yaxis_from_corners(gt_corners, self.device)
        else:
            gt_bbox = tensor_bbox_from_corners(gt_corners, self.device)

        pose_diff, per_diff = eval_part_full(feed_dict['gt_part'], pred_dict['part'],
                                             per_instance=per_instance, yaxis_only=self.sym)
        init_part_pose = feed_dict['state']['part']
        init_pose_diff, init_per_diff = eval_part_full(feed_dict['gt_part'], init_part_pose,
                                                       per_instance=per_instance,
                                                       yaxis_only=self.sym)
        loss_dict.update(pose_diff)
        loss_dict.update({f'init_{key}': value for key, value in init_pose_diff.items()})
        if per_instance:
            per_diff.update({f'init_{key}': value for key, value in init_per_diff.items()})
            self.record_per_diff(feed_dict, {'test': per_diff})

        loss_dict.update(compute_part_dof_loss(feed_dict['gt_part'], pred_dict['part'],
                                               self.pose_loss_type))

        corner_loss, corner_per_diff = compute_point_pose_loss(feed_dict['gt_part'], pred_dict['part'],
                                                               gt_bbox,
                                                               metric=self.pose_loss_type['point'])
        loss_dict['corner_loss'] = corner_loss

        root_delta = feed_dict['root_delta']
        eye_mat = torch.cat([torch.eye(self.num_parts), torch.zeros(2, self.num_parts)], dim=0)
        part_mask = eye_mat[labels, ].to(labels.device).transpose(-1, -2)  # [B, P, N]

        def masked_mean(value, mask):
            return torch.sum(value * mask) / torch.clamp(torch.sum(mask), min=1.0)

        if 'point_rotation' in pred_dict:
            point_rotation = pred_dict['point_rotation']  # [B, P, N, 3, 3]
            gt_rotation = root_delta['rotation'].unsqueeze(-3)  # [B, P, 1, 3, 3]
            if point_rotation.shape[1] == 1 and len(point_rotation.shape) > len(gt_rotation.shape):
                point_rotation = point_rotation.squeeze(1)  # [B, N, 3, 3]
            if self.sym:
                rloss = rot_yaxis_loss(gt_rotation, point_rotation)
            else:
                rloss = rot_trace_loss(gt_rotation, point_rotation, metric=self.pose_loss_type['r'])
            loss_dict['rloss'] = masked_mean(rloss, part_mask)

        self.summarize_losses(loss_dict)

    def update(self):
        self.prepare_data(self.raw_feed_dict)
        self.pred_dict = self.net(self.feed_dict, test_mode=False)
        self.compute_loss(test_mode=False)
        self.loss_dict['total_loss'].backward()

    def test(self, save=False, no_eval=False, epoch=0):
        with torch.no_grad():
            self.prepare_data(self.raw_feed_dict)
            self.pred_dict = self.net(self.feed_dict, test_mode=True)
            self.compute_loss(test_mode=True, per_instance=save)


class EvalTrackModel(BaseModel):
    def __init__(self, cfg):
        super(EvalTrackModel, self).__init__(cfg)
        self.net = PartCanonNet(cfg)
        self.npcs_net = CoordNet(cfg)
        self.tree = cfg['obj_tree']
        self.root = [p for p in range(len(self.tree)) if self.tree[p] == -1][0]
        self.gt_init = cfg['init_frame']['gt']
        self.nocs_otf = 'nocs_otf' in cfg and cfg['nocs_otf']
        if self.nocs_otf:
            assert cfg['batch_size'] == 1
            self.meta_root = pjoin(cfg['root_dset'], 'render', 'real_test', cfg['obj_category'])
        self.radius = cfg['data_radius']
        self.cfg = cfg
        self.track_cfg = cfg['track_cfg']
        self.raw_feed_dict = {}
        self.npcs_feed_dict = []
        self.timer = Timer(True)

    def convert_init_frame_data(self, frame):
        feed_frame = {}
        for key, item in frame.items():
            if key not in ['meta', 'labels', 'points', 'nocs']:
                continue
            if key in ['meta']:
                pass
            elif key in ['labels']:
                item = item.long().to(self.device)
            else:
                item = item.float().to(self.device)
            feed_frame[key] = item
        gt_part = part_model_batch_to_part(cvt_torch(frame['meta']['nocs2camera'], self.device), self.num_parts,
                                           self.cfg['device'])
        feed_frame.update({'gt_part': gt_part})

        return feed_frame

    def convert_subseq_frame_data(self, data):
        gt_part = part_model_batch_to_part(cvt_torch(data['meta']['nocs2camera'], self.device), self.num_parts,
                                           self.cfg['device'])
        input = {'points': data['points'],
                 'points_mean': data['meta']['points_mean'],
                 'gt_part': gt_part}

        if 'nocs' in data:
            input['npcs'] = data['nocs']
        input = cvt_torch(input, self.device)
        input['meta'] = data['meta']
        if 'labels' in data:
            input['labels'] = data['labels'].long().to(self.device)
        return input

    def convert_subseq_frame_npcs_data(self, data):
        input = {}
        for key, item in data.items():
            if key not in ['meta', 'labels', 'points', 'nocs']:
                continue
            elif key in ['meta']:
                pass
            elif key in ['labels']:
                item = item.long().to(self.device)
            else:
                item = item.float().to(self.device)
            input[key] = item
        input['points_mean'] = data['meta']['points_mean'].float().to(self.device)
        return input

    def set_data(self, data):
        self.feed_dict = []
        self.npcs_feed_dict = []
        for i, frame in enumerate(data):
            if i == 0:
                self.feed_dict.append(self.convert_init_frame_data(frame))
            else:
                self.feed_dict.append(self.convert_subseq_frame_data(frame))
            self.npcs_feed_dict.append(self.convert_subseq_frame_npcs_data(frame))

    def forward(self, save=False):
        self.timer.tick()

        pred_poses = []
        gt_part = self.feed_dict[0]['gt_part']
        if self.gt_init:
            pred_poses.append(gt_part)
        else:
            part = add_noise_to_part_dof(gt_part, self.pose_perturb_cfg)
            if 'crop_pose' in self.feed_dict[0]['meta']:
                crop_pose = part_model_batch_to_part(cvt_torch(self.feed_dict[0]['meta']['crop_pose'], self.device),
                                                     self.num_parts, self.device)
                for key in ['translation', 'scale']:
                    part[key] = crop_pose[key]
            pred_poses.append(part)

        self.timer.tick()

        time_dict = {'crop': 0.0, 'npcs_net': 0.0, 'rot_all': 0.0}

        frame_nums = []
        npcs_pred = []
        with torch.no_grad():
            for i, input in enumerate(self.feed_dict):
                frame_nums.append([path.split('.')[-2].split('/')[-1] for path in input['meta']['path']])
                if i == 0:
                    npcs_pred.append(None)
                    continue
                perturbed_part = add_noise_to_part_dof(self.feed_dict[i - 1]['gt_part'], self.pose_perturb_cfg)
                if 'crop_pose' in self.feed_dict[i]['meta']:
                    crop_pose = part_model_batch_to_part(
                        cvt_torch(self.feed_dict[i]['meta']['crop_pose'], self.device),
                        self.num_parts, self.device)
                    for key in ['translation', 'scale']:
                        perturbed_part[key] = crop_pose[key]

                last_pose = {key: value.clone() for key, value in pred_poses[-1].items()}

                self.timer.tick()
                if self.nocs_otf:
                    center = last_pose['translation'].reshape(3).detach().cpu().numpy()  # [3]
                    scale = last_pose['scale'].reshape(1).detach().cpu().item()
                    depth_path = input['meta']['ori_path'][0]
                    category, instance = input['meta']['path'][0].split('/')[-4:-2]
                    pre_fetched = input['meta']['pre_fetched']
                    pre_fetched = {key: value.reshape(value.shape[1:]) for key, value in pre_fetched.items()}

                    pose = {key: value.squeeze(0).squeeze(0).detach().cpu().numpy() for key, value in
                            input['gt_part'].items()}
                    full_data = full_data_from_depth_image(depth_path, category, instance, center, self.radius * scale, pose,
                                                           num_points=input['points'].shape[-1], device=self.device,
                                                           mask_from_nocs2d=self.track_cfg['nocs2d_label'],
                                                           nocs2d_path=self.track_cfg['nocs2d_path'],
                                                           pre_fetched=pre_fetched)

                    points, nocs, labels = full_data['points'], full_data['nocs'], full_data['labels']

                    points = cvt_torch(points, self.device)
                    points -= self.npcs_feed_dict[i]['points_mean'].reshape(1, 3)
                    input['points'] = points.transpose(-1, -2).reshape(1, 3, -1)
                    input['labels'] = torch.tensor(full_data['labels']).to(self.device).long().reshape(1, -1)
                    nocs = cvt_torch(nocs, self.device)
                    self.npcs_feed_dict[i]['points'] = input['points']
                    self.npcs_feed_dict[i]['labels'] = input['labels']
                    self.npcs_feed_dict[i]['nocs'] = nocs.transpose(-1, -2).reshape(1, 3, -1)

                    time_dict['crop'] += self.timer.tick()

                state = {'part': last_pose}
                input['state'] = state

                npcs_canon_pose = {key: last_pose[key][:, self.root].clone() for key in
                                   ['rotation', 'translation', 'scale']}
                npcs_input = self.npcs_feed_dict[i]
                npcs_input['canon_pose'] = npcs_canon_pose
                npcs_input['init_part'] = last_pose
                cur_npcs_pred = self.npcs_net(npcs_input)  # seg: [B, P, N], npcs: [B, P * 3, N]
                npcs_pred.append(cur_npcs_pred)
                pred_npcs, pred_seg = cur_npcs_pred['nocs'], cur_npcs_pred['seg']
                pred_npcs = pred_npcs.reshape(len(pred_npcs), self.num_parts, 3, -1)  # [B, P, 3, N]
                pred_labels = torch.max(pred_seg, dim=-2)[1]  # [B, P, N] -> [B, N]

                time_dict['npcs_net'] += self.timer.tick()

                input['pred_labels'], input['pred_nocs'] = pred_labels, pred_npcs
                input['pred_label_conf'] = pred_seg[:, 0]  # [B, P, N]
                if self.track_cfg['gt_label'] or self.track_cfg['nocs2d_label']:
                    input['pred_labels'] = npcs_input['labels']

                pred_dict = self.net(input, test_mode=True)
                pred_poses.append(pred_dict['part'])

                time_dict['rot_all'] += self.timer.tick()

        self.pred_dict = {'poses': pred_poses, 'npcs_pred': npcs_pred}

        if save:
            gt_corners = self.feed_dict[0]['meta']['nocs_corners'].cpu().numpy()
            corner_list = []
            for i, pred_pose in enumerate(self.pred_dict['poses']):
                if i == 0:
                    corner_list.append(None)
                    continue
                pred_labels = torch.max(self.pred_dict['npcs_pred'][i]['seg'], dim=-2)[1]  # [B, P, N] -> [B, N]
                pred_nocs = choose_coord_by_label(self.pred_dict['npcs_pred'][i]['nocs'].transpose(-1, -2),
                                                  pred_labels)
                pred_corners = get_pred_nocs_corners(pred_labels, pred_nocs, self.num_parts)
                corner_list.append(pred_corners)

            gt_poses = [{key: value.detach().cpu().numpy() for key, value in frame[f'gt_part'].items()}
                        for frame in self.feed_dict]
            save_dict = {'pred': {'poses': [{key: value.detach().cpu().numpy() for key, value in pred_pose.items()}
                                            for pred_pose in pred_poses],
                                  'corners': corner_list},
                         'gt': {'poses': gt_poses, 'corners': gt_corners},
                         'frame_nums': frame_nums}

            save_path = pjoin(self.cfg['experiment_dir'], 'results', 'data')
            ensure_dirs([save_path])
            for i, path in enumerate(self.feed_dict[0]['meta']['path']):
                instance, track_num = path.split('.')[-2].split('/')[-3: -1]
                with open(pjoin(save_path, f'{instance}_{track_num}.pkl'), 'wb') as f:
                    cur_dict = get_ith_from_batch(save_dict, i, to_single=False)
                    pickle.dump(cur_dict, f)

    def compute_loss(self, test=False, per_instance=False, eval_iou=False, test_prefix=None):
        feed_dict = self.feed_dict
        pred_dict = self.pred_dict
        loss_dict = {}

        avg_pose_diff, all_pose_diff = {}, {}
        avg_init_diff, all_init_diff = {}, {}
        avg_iou, all_iou = {}, {}
        avg_seg_loss, all_seg_loss = [], {}
        avg_nocs_loss, all_nocs_loss = [], {}
        gt_corners = feed_dict[0]['meta']['nocs_corners'].float().to(self.device)

        for i, pred_pose in enumerate(pred_dict['poses']):

            pose_diff, per_diff = eval_part_full(feed_dict[i]['gt_part'], pred_pose,
                                                 per_instance=per_instance, yaxis_only=self.sym)

            if i > 0:
                add_dict(avg_pose_diff, pose_diff)
                if per_instance:
                    self.record_per_diff(feed_dict[i], per_diff)

            all_pose_diff[i] = deepcopy(pose_diff)

            if i > 0:
                init_pose_diff, init_per_diff = eval_part_full(feed_dict[i]['gt_part'], pred_dict['poses'][i - 1],
                                                               per_instance=per_instance,
                                                               yaxis_only=self.sym)

                add_dict(avg_init_diff, init_pose_diff)
                all_init_diff[i] = deepcopy(init_pose_diff)

            if i > 0:
                if 'labels' in self.npcs_feed_dict[i]:
                    seg_loss = compute_miou_loss(pred_dict['npcs_pred'][i]['seg'], self.npcs_feed_dict[i]['labels'],
                                                 per_instance=False)
                    avg_seg_loss.append(seg_loss)
                    all_seg_loss[i] = seg_loss

                pred_labels = torch.max(pred_dict['npcs_pred'][i]['seg'], dim=-2)[1]  # [B, P, N] -> [B, N]

                if 'nocs' in self.npcs_feed_dict[i]:
                    nocs_loss = compute_nocs_loss(pred_dict['npcs_pred'][i]['nocs'], self.npcs_feed_dict[i]['nocs'],
                                                  labels=pred_labels,
                                                  confidence=None, loss='l2', self_supervise=False,
                                                  per_instance=False)
                    avg_nocs_loss.append(nocs_loss)
                    all_nocs_loss[i] = nocs_loss

                pred_nocs = choose_coord_by_label(pred_dict['npcs_pred'][i]['nocs'].transpose(-1, -2),
                                                  pred_labels)

                if eval_iou:
                    pred_corners = get_pred_nocs_corners(pred_labels, pred_nocs, self.num_parts)
                    pred_corners = torch.tensor(pred_corners).to(self.device).float()

                    def calc_iou(gt_pose, pred_pose):
                        iou, per_iou = eval_single_part_iou(gt_corners, pred_corners, gt_pose, pred_pose,
                                                            separate='both', nocs=self.nocs_otf, sym=self.sym)

                        return iou, per_iou

                    iou, per_iou = calc_iou(feed_dict[i]['gt_part'], pred_pose)
                    add_dict(avg_iou, iou)
                    if per_instance:
                        self.record_per_diff(feed_dict[i], per_iou)
                    all_iou[i] = deepcopy(iou)

        avg_pose_diff = divide_dict(avg_pose_diff, len(pred_dict['poses']) - 1)
        avg_init_diff = divide_dict(avg_init_diff, len(pred_dict['poses']) - 1)
        loss_dict.update({'avg_pred': avg_pose_diff, 'avg_init': avg_init_diff,
                          'frame_pred': all_pose_diff, 'frame_init': all_init_diff})
        if len(avg_seg_loss) > 0:
            avg_seg_loss = torch.mean(torch.stack(avg_seg_loss))
            loss_dict.update({'avg_seg': avg_seg_loss, 'frame_seg': all_seg_loss})
        if len(avg_nocs_loss) > 0:
            avg_nocs_loss = torch.mean(torch.stack(avg_nocs_loss))
            loss_dict.update({'avg_nocs': avg_nocs_loss, 'frame_seg': all_nocs_loss})
        if eval_iou:
            avg_iou = divide_dict(avg_iou, len(pred_dict['poses']) - 1)
            loss_dict.update({'avg_iou': avg_iou, 'frame_iou': all_iou})

        self.loss_dict = loss_dict

    def test(self, save=False, no_eval=False, epoch=0):
        self.forward(save=save)
        if not no_eval:
            self.compute_loss(test=True, per_instance=save, eval_iou=True, test_prefix='test')
        else:
            self.loss_dict = {}


