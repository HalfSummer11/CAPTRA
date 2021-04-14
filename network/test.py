import argparse
import os
import torch
import pickle
import logging
import sys
import time
from tqdm import tqdm
import numpy as np
from copy import deepcopy
from os.path import join as pjoin

base_dir = os.path.dirname(__file__)
sys.path.append(base_dir)
sys.path.append(os.path.join(base_dir, '..'))

from utils import boolean_string, ensure_dirs, add_dict, log_loss_summary, print_composite
from data.dataset import get_dataloader
from configs.config import get_config
from trainer import Trainer
from parse_args import add_args


def parse_args():
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    parser.add_argument('--num_iter', default=1, type=int)
    parser.add_argument('--mode_name', default='test')

    return parser.parse_args()


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    cfg = get_config(args, save=False)

    '''LOG'''
    log_dir = pjoin(cfg['experiment_dir'], 'log')
    ensure_dirs(log_dir)

    logger = logging.getLogger("TestModel")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/log_test.txt' % (log_dir))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(cfg)

    '''TRAINER'''
    trainer = Trainer(cfg, logger)
    trainer.resume()

    '''testing'''
    save = cfg['save']
    no_eval = cfg['no_eval']

    dataset_name = args.mode_name

    test_dataloader = get_dataloader(cfg, dataset_name)
    test_loss = {'cnt': 0}

    zero_time = time.time()
    time_dict = {'data_proc': 0.0, 'network': 0.0}
    total_frames = 0

    for i, data in tqdm(enumerate(test_dataloader), total=len(test_dataloader), smoothing=0.9):
        num_frames = len(data)
        total_frames += num_frames
        print(f'Trajectory {i}, {num_frames:8} frames****************************')

        start_time = time.time()
        elapse = start_time - zero_time
        time_dict['data_proc'] += elapse
        print(f'Data Preprocessing: {elapse:8.2f}s {num_frames / elapse:8.2f}FPS')

        pred_dict, loss_dict = trainer.test(data, save=save, no_eval=no_eval)

        elapse = time.time() - start_time
        time_dict['network'] += elapse
        print(f'Network Forwarding: {elapse:8.2f}s {num_frames / elapse:8.2f}FPS')

        loss_dict['cnt'] = 1
        add_dict(test_loss, loss_dict)

        zero_time = time.time()

    print(f'Overall, {total_frames:8} frames****************************')
    print(f'Data Preprocessing: {time_dict["data_proc"]:8.2f}s {total_frames / time_dict["data_proc"]:8.2f}FPS')
    print(f'Network Forwarding: {time_dict["network"]:8.2f}s {total_frames / time_dict["network"]:8.2f}FPS')
    if cfg['batch_size'] > 1:
        print(f'PLEASE SET batch_size = 1 TO TEST THE SPEED. CURRENT BATCH_SIZE: cfg["batch_size"]')

    cnt = test_loss.pop('cnt')
    log_loss_summary(test_loss, cnt, lambda x, y: log_string('Test {} is {}'.format(x, y)))
    if save and not no_eval:
        trainer.model.save_per_diff()


if __name__ == '__main__':
    args = parse_args()
    torch.multiprocessing.set_start_method('spawn')
    main(args)

