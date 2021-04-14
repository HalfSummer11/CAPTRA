import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os, sys
from os.path import join as pjoin
BASEPATH = os.path.dirname(__file__)
sys.path.insert(0, pjoin(BASEPATH, '..', '..'))
from configs.config import get_config
from pose_utils.rotations import normalize_vector, compute_rotation_matrix_from_ortho6d


class Upsample(nn.Module):
    def __init__(self, scale_factor, mode='nearest'):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor,
                             mode=self.mode)


def get_conv_layer(kernel_size, in_channels, out_channels, stride=1, pad_type='valid', use_bias=True):
    """
    returns a list of [pad, conv] => should be += to some list, then apply sequential
    """
    conv = nn.Conv1d(in_channels, out_channels,
                     kernel_size=kernel_size,
                     stride=stride, bias=use_bias)

    if pad_type == 'valid':
        return [conv]

    def ZeroPad1d(sizes):
        return nn.ConstantPad1d(sizes, 0)

    if pad_type == 'reflect':
        pad = nn.ReflectionPad1d
    elif pad_type == 'replicate':
        pad = nn.ReplicationPad1d
    elif pad_type == 'zero':
        pad = ZeroPad1d
    else:
        assert 0, "Unsupported padding type: {}".format(pad_type)

    pad_l = (kernel_size - 1) // 2
    pad_r = kernel_size - 1 - pad_l
    return [pad((pad_l, pad_r)), conv]


def get_acti_layer(acti='relu', inplace=True):
    if acti == 'relu':
        return [nn.ReLU(inplace=inplace)]
    elif acti == 'lrelu':
        return [nn.LeakyReLU(0.2, inplace=inplace)]
    elif acti == 'tanh':
        return [nn.Tanh()]
    elif acti == 'sigmoid':
        return [nn.Sigmoid()]
    elif acti == 'softplus':
        return [nn.Softplus()]
    elif acti == 'none':
        return []
    else:
        assert 0, "Unsupported activation: {}".format(acti)


def get_norm_layer(norm='none', norm_dim=None, channel_per_group=2):
    if norm == 'bn':
        return [nn.BatchNorm1d(norm_dim)]
    elif norm == 'gn':
        return [nn.GroupNorm(norm_dim // channel_per_group, norm_dim)]
    elif norm == 'in':
        return [nn.InstanceNorm1d(norm_dim, affine=True)]
    elif norm == 'none':
        return []
    else:
        assert 0, "Unsupported normalization: {}".format(norm)


def get_dropout_layer(dropout=None):
    if dropout is not None:
        return [nn.Dropout(p=dropout)]
    else:
        return []


def get_conv_block(kernel_size, in_channels, out_channels, stride=1, pad_type='valid', use_bias=True, inplace=True,
                   dropout=None, norm='none', acti='none', acti_first=False):
    """
    returns a list of [pad, conv, norm, acti] or [acti, pad, conv, norm]
    """

    layers = get_conv_layer(kernel_size, in_channels, out_channels, stride=stride, pad_type=pad_type, use_bias=use_bias)
    layers += get_dropout_layer(dropout)
    layers += get_norm_layer(norm, norm_dim=out_channels)
    acti_layers = get_acti_layer(acti, inplace=inplace)

    if acti_first:
        return acti_layers + layers
    else:
        return layers + acti_layers


def get_linear_block(in_dim, out_dim, dropout=None, norm='none', acti='relu'):

    use_bias = True
    layers = []
    layers.append(nn.Linear(in_dim, out_dim, bias=use_bias))
    layers += get_dropout_layer(dropout)
    layers += get_norm_layer(norm, norm_dim=out_dim)
    layers += get_acti_layer(acti)

    return layers


def get_point_mlp(in_dim, out_dim, dims, acti='none', dropout=True, last_bn=False,
                  keep_list=False):
    """
    convs w/ kernel_size = 1
    By default,
    - use_bias=True;
    - intermediate layers has batch norm + dropout(0.5), acti=relu
    - the last layer has acti=acti
    """
    dropout = 0.5 if dropout else None
    layers = []
    dims = [in_dim] + dims + [out_dim]
    for i in range(len(dims) - 2):
        layers += get_conv_block(1, dims[i], dims[i + 1],
                                 dropout=dropout, norm='bn', acti='relu')
    layers += get_conv_block(1, dims[-2], dims[-1], norm='bn' if last_bn else 'none', acti=acti)

    return layers if keep_list else nn.Sequential(*layers)


def get_activation(acti):
    if acti == "relu":
        return nn.ReLU()
    elif acti == 'lrelu':
        return nn.LeakyReLU()
    else:
        assert 0, f'Unsupported activation type {acti}'


class MLPConv1d(nn.Module):
    def __init__(self, in_channel, mlp, bn=True, gn=False, activation="relu", last_activation='none'):
        super(MLPConv1d, self).__init__()

        norm = 'gn' if gn else ('bn' if bn else 'none')

        layers = []
        last_channel = in_channel
        for i, out_channel in enumerate(mlp):
            last_layer = (i == len(mlp) - 1)
            norm = norm if not last_layer else 'none'
            acti = last_activation if last_layer else activation
            layers += get_conv_block(1, last_channel, out_channel, norm=norm, acti=acti)
            last_channel = out_channel
        self.model = nn.Sequential(*layers)
        self.out_channel = last_channel

    def forward(self, input):  # input: [B, in_channel, 1]
        return self.model(input)  # [B, out_channel, 1]


class RotationRegressor(nn.Module):
    def __init__(self, in_dim, num_parts, symmetric=False):
        super(RotationRegressor, self).__init__()
        gn = True
        rot_dim = 3 if symmetric else 6
        self.sym = symmetric
        mlp = [512, 512, 256]
        heads = num_parts
        self.rtvec_head = nn.ModuleList([MLPConv1d(in_dim, mlp + [rot_dim], bn=True, gn=gn, last_activation='none')
                                         for p in range(heads)])

        self.num_parts = num_parts

    def forward(self, feat):  # feat: [B, in_dim, 1/N]
        rtvec = torch.stack([rtvec_head(feat) for rtvec_head in self.rtvec_head], dim=1)  # [B, P, R, N]
        if not self.sym:
            raw_6d = rtvec.transpose(-1, -2)  # [B, (P,) 6, N] -> [B, (P,) N, 6]
            original_shape = raw_6d.shape
            rot_matrix = compute_rotation_matrix_from_ortho6d(raw_6d.reshape(-1, 6))
            rtvec = rot_matrix.reshape(original_shape[:-1] + (-1,)).transpose(-1, -2)
        else:
            raw_3d = rtvec.transpose(-1, -2)  # [B, (P,) 3, N] -> [B, (P,) N, 3]
            original_shape = raw_3d.shape
            norm_3d = normalize_vector(raw_3d.reshape(-1, 3))
            rtvec = norm_3d.reshape(original_shape).transpose(-1, -2)
        return rtvec


def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--config', type=str, default='config.yml', help='path to config.yml')
    parser.add_argument('--obj_config', type=str, default=None, help='path to obj_config.yml')
    parser.add_argument('--experiment_dir', type=str, default=None, help='root dir for all outputs')
    parser.add_argument('--show_fig', action='store_true', default=False, help='show figure')
    parser.add_argument('--save_fig', action='store_true', default=False, help='save figure')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    cfg = get_config(args, save=False)

