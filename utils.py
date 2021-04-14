import os
import torch
import time
import numpy as np
import random
import gc
import operator as op
from functools import reduce

from copy import deepcopy


def boolean_string(s):
    if s is None:
        return None
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


def update_dict(old_dict, new_dict):
    for key, value in new_dict.items():
        if isinstance(value, dict):
            if not key in old_dict:
                old_dict[key] = {}
            update_dict(old_dict[key], value)
        else:
            old_dict[key] = value


def merge_dict(old_dict, new_dict, init_val=[], f=lambda x, y: x + [y]):
    for key, value in new_dict.items():
        if not key in old_dict.keys():
            old_dict[key] = init_val
        old_dict[key] = f(old_dict[key], value)


def detach_dict(d):
    for key, value in d.items():
        if isinstance(value, dict):
            detach_dict(value)
        elif isinstance(value, torch.Tensor):
            d[key] = value.detach().cpu().numpy()


def add_dict(old_dict, new_dict):
    def copy_dict(d):
        ret = {}
        for key, value in d.items():
            if isinstance(value, dict):
                ret[key] = copy_dict(value)
            else:
                ret[key] = value
        del d
        return ret
    detach_dict(new_dict)
    for key, value in new_dict.items():
        if not key in old_dict.keys():
            if isinstance(value, dict):
                old_dict[key] = copy_dict(value)
            else:
                old_dict[key] = value
        else:
            if isinstance(value, dict):
                add_dict(old_dict[key], value)
            else:
                old_dict[key] += value


def ensure_dir(path, verbose=False):
    if not os.path.exists(path):
        if verbose:
            print("Create folder ", path)
        os.makedirs(path)
    else:
        if verbose:
            print(path, " already exists.")


def ensure_dirs(paths):
    if isinstance(paths, list):
        for path in paths:
            ensure_dir(path)
    else:
        ensure_dir(paths)


def write_loss(it, loss_dict, writer):
    def write_dict(d, prefix=None):
        for key, value in d.items():
            name = str(key) if prefix is None else '/'.join([prefix, str(key)])
            if isinstance(value, dict):
                write_dict(value, name)
            else:
                writer.add_scalar(name, value, it + 1)
    write_dict(loss_dict)


def log_loss_summary(loss_dict, cnt, log_loss):
    def log_dict(d, prefix=None):
        for key, value in d.items():
            name = str(key) if prefix is None else '/'.join([prefix, str(key)])
            if isinstance(value, dict):
                log_dict(value, name)
            else:
                log_loss(name, d[key] / cnt)
    log_dict(loss_dict)


def divide_dict(ddd, cnt):
    def div_dict(d):
        ret = {}
        for key, value in d.items():
            if isinstance(value, dict):
                ret[key] = div_dict(value)
            else:
                ret[key] = value / cnt
        return ret
    return div_dict(ddd)


def print_composite(data, beg=""):
    if isinstance(data, dict):
        print(f'{beg} dict, size = {len(data)}')
        for key, value in data.items():
            print(f'  {beg}{key}:')
            print_composite(value, beg + "    ")
    elif isinstance(data, list):
        print(f'{beg} list, len = {len(data)}')
        for i, item in enumerate(data):
            print(f'  {beg}item {i}')
            print_composite(item, beg + "    ")
    elif isinstance(data, np.ndarray) or isinstance(data, torch.Tensor):
        print(f'{beg} array of size {data.shape}')
    else:
        print(f'{beg} {data}')


class Timer:
    def __init__(self, on):
        self.on = on
        self.cur = time.time()

    def tick(self, str=None):
        if not self.on:
            return
        cur = time.time()
        diff = cur - self.cur
        self.cur = cur
        if str is not None:
            print(str, diff)
        return diff


def get_ith_from_batch(data, i, to_single=True):
    if isinstance(data, dict):
        return {key: get_ith_from_batch(value, i, to_single) for key, value in data.items()}
    elif isinstance(data, list):
        return [get_ith_from_batch(item, i, to_single) for item in data]
    elif isinstance(data, torch.Tensor):
        if to_single:
            return data[i].detach().cpu().item()
        else:
            return data[i].detach().cpu()
    elif isinstance(data, np.ndarray):
        return data[i]
    elif data is None:
        return None
    elif isinstance(data, str):
        return data
    else:
        assert 0, f'Unsupported data type {type(data)}'


def cvt_torch(x, device):
    if isinstance(x, np.ndarray):
        return torch.tensor(x).float().to(device)
    elif isinstance(x, torch.Tensor):
        return x.float().to(device)
    elif isinstance(x, dict):
        return {key: cvt_torch(value, device) for key, value in x.items()}
    elif isinstance(x, list):
        return [cvt_torch(item, device) for item in x]
    elif x is None:
        return None


class Mixture:
    def __init__(self, proportion_dict):
        self.keys = list(proportion_dict.keys())
        self.cumsum = np.cumsum([proportion_dict[key] for key in self.keys])
        assert self.cumsum[-1] == 1.0, 'Proportions do not sum to one'

    def sample(self):
        choice = random.random()
        idx = np.searchsorted(self.cumsum, choice)
        return self.keys[idx]


def inspect_tensors(verbose=False):
    total = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                tmp = reduce(op.mul, obj.size())
                total += tmp
                if verbose and obj.size() == (1, 128, 128) or obj.size() == (1, 256, 256) or obj.size() == (1, 128, 256) or obj.size() == (1, 256, 128):
                    print(obj.size(), obj)
                    # print(type(obj), obj. tmp, obj.size())
        except:
            pass
    print("=================== Total = {} ====================".format(total))


def eyes_like(tensor: torch.Tensor):  # [Bs, 3, 3]
    assert tensor.shape[-2:] == (3, 3), 'eyes must be applied to tensor w/ last two dims = (3, 3)'
    eyes = torch.eye(3, dtype=tensor.dtype, device=tensor.device)
    eyes = eyes.reshape(tuple(1 for _ in range(len(tensor.shape) - 2)) + (3, 3)).repeat(tensor.shape[:-2] + (1, 1))
    return eyes


def flatten_dict(loss_dict):
    def flatten_d(d, prefix=None):
        new_d = {}
        for key, value in d.items():
            name = str(key) if prefix is None else '/'.join([prefix, str(key)])
            if isinstance(value, dict):
                new_d.update(flatten_d(value, name))
            else:
                new_d[name] = value
        return new_d
    ret = flatten_d(loss_dict)
    return ret


def per_dict_to_csv(loss_dict, csv_name):
    all_ins = {inst: flatten_dict(loss_dict[inst]) for inst in loss_dict}

    keys = list(list(all_ins.values())[0].keys())
    dir = os.path.dirname(csv_name)
    if not os.path.exists(dir):
        os.makedirs(dir)
    with open(csv_name, 'w') as f:
        def fprint(s):
            print(s, end='', file=f)
        for key in keys:
            fprint(f',{key}')
        fprint('\n')
        for inst in all_ins:
            fprint(f'{inst}')
            for key in keys:
                fprint(f',{all_ins[inst][key]}')
            fprint('\n')
