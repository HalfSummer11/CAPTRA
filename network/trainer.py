import argparse
import os
import logging
import math

from os.path import join as pjoin

from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.optim import lr_scheduler
from models.model import CanonCoordModel, RotationModel, EvalTrackModel

from utils import update_dict, ensure_dirs


def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find(
                'Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'gaussian':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
    return init_fun


def get_scheduler(optimizer, cfg, it=-1):
    scheduler = None
    if optimizer is None:
        return scheduler
    if 'lr_policy' not in cfg or cfg['lr_policy'] == 'constant':
        scheduler = None  # constant scheduler
    elif cfg['lr_policy'] == 'step':
        scheduler = lr_scheduler.StepLR(optimizer,
                                        step_size=cfg['lr_step_size'],
                                        gamma=cfg['lr_gamma'],
                                        last_epoch=it)
    else:
        assert 0, '{} not implemented'.format(cfg['lr_policy'])
    return scheduler


def get_optimizer(params, cfg):
    if len(params) == 0:
        return None
    if cfg['optimizer'] == 'Adam':
        optimizer = torch.optim.Adam(
            params, lr=cfg['learning_rate'],
            betas=(0.9, 0.999), eps=1e-08,
            weight_decay=cfg['weight_decay'])
    elif cfg['optimizer'] == 'SGD':
        optimizer = torch.optim.SGD(
            params, lr=cfg['learning_rate'],
            momentum=0.9)
    else:
        assert 0, "Unsupported optimizer type {}".format(cfg['optimizer'])
    return optimizer


def get_last_model(dirname, key=""):
    if not os.path.exists(dirname):
        return None
    models = [pjoin(dirname, f) for f in os.listdir(dirname) if
              os.path.isfile(pjoin(dirname, f)) and
              key in f and ".pt" in f]
    if models is None or len(models) == 0:
        return None
    models.sort()
    last_model_name = models[-1]
    return last_model_name


class Trainer(nn.Module):
    def __init__(self, cfg, logger=None):
        super(Trainer, self).__init__()
        self.ckpt_dir = pjoin(cfg['experiment_dir'], 'ckpt')
        ensure_dirs(self.ckpt_dir)

        self.device = cfg['device']
        if cfg['network']['type'] == 'canon_coord':
            self.model = CanonCoordModel(cfg)
        elif cfg['network']['type'] == 'rot':
            self.model = RotationModel(cfg)
        elif cfg['network']['type'] == 'rot_coord_track':
            self.model = EvalTrackModel(cfg)

        self.network_type = cfg['network']['type']

        if self.network_type in ['rot_coord_track']:
            self.coord_exp_dir = pjoin(cfg['coord_exp']['dir'], 'ckpt')
            self.coord_resume_epoch = cfg['coord_exp']['resume_epoch']
        else:
            self.coord_exp_dir = None

        self.optimizer = get_optimizer([p for p in self.model.parameters() if p.requires_grad], cfg)
        self.scheduler = get_scheduler(self.optimizer, cfg)

        self.apply(weights_init(cfg['weight_init']))
        self.epoch = 0
        self.iteration = 0
        self.loss_dict = {}
        self.cfg = cfg
        self.logger = logger

        self.to(self.device)

    def log_string(self, str):
        print(str)
        if self.logger is not None:
            self.logger.info(str)

    def step_epoch(self):
        cfg = self.cfg
        self.epoch += 1

        if self.scheduler is not None and self.scheduler.get_lr()[0] > cfg['lr_clip']:
            self.scheduler.step()
        self.lr = self.scheduler.get_lr()[0]
        self.log_string("Epoch %d/%d, learning rate = %f" % (
            self.epoch, cfg['total_epoch'], self.lr))

        momentum = cfg['momentum_original'] * (
                cfg['momentum_decay'] ** (self.epoch // cfg['momentum_step_size']))
        momentum = max(momentum, cfg['momentum_min'])
        self.log_string("BN momentum updated to %f" % momentum)
        self.momentum = momentum

        def bn_momentum_adjust(m, momentum):
            if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
                m.momentum = momentum

        self.model = self.model.apply(lambda x: bn_momentum_adjust(x, momentum))

    def resume(self):
        def get_model(dir, resume_epoch):
            last_model_name = get_last_model(dir)
            print('last model name', last_model_name)
            if resume_epoch > 0:
                specified_model = pjoin(dir, f"model_{resume_epoch:04d}.pt")
                if os.path.exists(specified_model):
                    last_model_name = specified_model
            return last_model_name

        ckpt = OrderedDict()

        if self.coord_exp_dir is not None:
            coord_name = get_model(self.coord_exp_dir, self.coord_resume_epoch)
            if coord_name is None:
                assert 0, 'Invalid CoordNet dir'
            else:
                print(f'Load CoordNet model from {coord_name}')
            coord_state_dict = torch.load(coord_name, map_location=self.device)['model']
            coord_keys = list(coord_state_dict.keys())
            for key in coord_keys:
                if key.startswith('net'):
                    ckpt['npcs_net' + key[3:]] = coord_state_dict[key]

        model_name = get_model(self.ckpt_dir, self.cfg['resume_epoch'])

        if model_name is None:
            self.log_string('Initialize from 0')
        else:
            state_dict = torch.load(model_name, map_location=self.device)
            self.epoch = state_dict['epoch']
            self.iteration = state_dict['iteration']
            ckpt.update(state_dict['model'])

            if self.optimizer is not None:
                try:
                    self.optimizer.load_state_dict(state_dict['optimizer'])
                except (ValueError, KeyError):
                    pass  # when new params are added, just give up on the old states
                self.scheduler = get_scheduler(self.optimizer, self.cfg, self.epoch)

            self.log_string('Resume from epoch %d' % self.epoch)

        self.model.load_state_dict(ckpt, strict=False)

        print(self.model)

        return self.epoch

    def save(self, name=None, extra_info=None):
        epoch = self.epoch
        if name is None:
            name = f'model_{epoch:04d}'
        savepath = pjoin(self.ckpt_dir, "%s.pt" % name)
        state = {
            'epoch': epoch,
            'iteration': self.iteration,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        if isinstance(extra_info, dict):
            update_dict(state, extra_info)
        torch.save(state, savepath)
        self.log_string("Saving model at epoch {}, path {}".format(epoch, savepath))

    def update(self, data):
        self.optimizer.zero_grad()
        self.model.train()
        self.model.set_data(data)
        self.model.update()
        loss_dict = self.model.loss_dict
        update_dict(self.loss_dict, loss_dict)
        self.optimizer.step()
        self.iteration += 1
        return loss_dict

    def test(self, data, save=False, no_eval=False):
        self.model.eval()
        self.model.set_data(data)
        self.model.test(save=save, no_eval=no_eval, epoch=self.epoch)
        pred_dict = self.model.pred_dict
        loss_dict = self.model.loss_dict

        return pred_dict, loss_dict

