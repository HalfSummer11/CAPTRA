import argparse
import os
import torch
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

from utils import boolean_string, ensure_dirs, write_loss, add_dict, log_loss_summary, print_composite
from data.dataset import get_dataloader
from configs.config import get_config
from trainer import Trainer
from parse_args import add_args

def parse_args():
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    parser.add_argument('--real_only', action='store_true')
    parser.add_argument('--syn_only', action='store_true')
    parser.add_argument('--syn_n', type=int, default=1)
    parser.add_argument('--val_name', type=str, default='real_test')
    parser.add_argument('--downsample', type=int, default=5)

    return parser.parse_args()


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    cfg = get_config(args)

    '''LOG'''
    log_dir = pjoin(cfg['experiment_dir'], 'log')
    ensure_dirs(log_dir)

    logger = logging.getLogger("TrainNOCSModel")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/log.txt' % (log_dir))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(cfg)

    '''DATA'''
    test_dataloader = get_dataloader(cfg, args.val_name, downsampling=args.downsample)

    if not args.syn_only:
        train_real_dataloader = get_dataloader(cfg, 'real_train', shuffle=True)
        syn_train_len = len(train_real_dataloader) * args.syn_n
    else:
        syn_train_len = args.syn_n

    train_syn_dataloader = get_dataloader(cfg, 'train', shuffle=True)
    syn_train_cycle = iter(train_syn_dataloader)
    num_div = len(train_syn_dataloader) // syn_train_len

    '''TRAINER'''
    trainer = Trainer(cfg, logger)
    start_epoch = trainer.resume()

    def test_all():
        '''testing'''
        test_loss = {}
        for i, data in enumerate(test_dataloader):
            pred_dict, loss_dict = trainer.test(data)
            loss_dict['cnt'] = 1
            add_dict(test_loss, loss_dict)

        cnt = test_loss.pop('cnt')
        log_loss_summary(test_loss, cnt, lambda x, y: log_string('real_test {} is {}'.format(x, y)))

    test_all()

    for epoch in range(start_epoch, cfg['total_epoch']):
        trainer.step_epoch()

        '''training'''
        if not args.real_only:
            train_loss = {}
            for i in range(syn_train_len):
                data = next(syn_train_cycle)
                loss_dict = trainer.update(data)
                loss_dict['cnt'] = 1
                add_dict(train_loss, loss_dict)

            cnt = train_loss.pop('cnt')
            log_loss_summary(train_loss, cnt, lambda x, y: log_string('Syn_Train {} is {}'.format(x, y)))

        if not args.syn_only:
            train_loss = {}
            for i, data in enumerate(train_real_dataloader):
                loss_dict = trainer.update(data)
                loss_dict['cnt'] = 1
                add_dict(train_loss, loss_dict)

            cnt = train_loss.pop('cnt')
            log_loss_summary(train_loss, cnt, lambda x, y: log_string('Real_Train {} is {}'.format(x, y)))

        if (epoch + 1) % cfg['freq']['save'] == 0:
            trainer.save()

        test_all()
        if (epoch + 1) % num_div == 0:
            syn_train_cycle = iter(train_syn_dataloader)


if __name__ == '__main__':
    args = parse_args()
    torch.multiprocessing.set_start_method('spawn')
    main(args)

