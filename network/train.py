import argparse
import os
import torch
import logging
import sys
import time
from os.path import join as pjoin

base_dir = os.path.dirname(__file__)
sys.path.append(base_dir)
sys.path.append(os.path.join(base_dir, '..'))

from utils import ensure_dirs, add_dict, log_loss_summary, print_composite
from data.dataset import get_dataloader
from configs.config import get_config
from trainer import Trainer
from parse_args import add_args


def parse_args():
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    parser.add_argument('--use_val', type=str, default=None)
    return parser.parse_args()


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    cfg = get_config(args)

    '''LOG'''
    log_dir = pjoin(cfg['experiment_dir'], 'log')
    ensure_dirs(log_dir)

    logger = logging.getLogger("TrainModel")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/log.txt' % (log_dir))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(cfg)

    '''DATA'''
    train_dataloader = get_dataloader(cfg, 'train', shuffle=True)
    test_dataloader = get_dataloader(cfg, 'test')

    if args.use_val is not None:
        val_dataloader = get_dataloader(cfg, args.use_val)
    else:
        val_dataloader = None

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
        log_loss_summary(test_loss, cnt, lambda x, y: log_string('Test {} is {}'.format(x, y)))

        if val_dataloader is not None:
            val_loss = {}
            for i, data in enumerate(val_dataloader):
                pred_dict, loss_dict = trainer.test(data)
                loss_dict['cnt'] = 1
                add_dict(val_loss, loss_dict)

            cnt = val_loss.pop('cnt')
            log_loss_summary(val_loss, cnt, lambda x, y: log_string('{} {} is {}'.format(args.use_val, x, y)))

    for epoch in range(start_epoch, cfg['total_epoch']):
        trainer.step_epoch()
        train_loss = {}

        '''training'''
        for i, data in enumerate(train_dataloader):
            loss_dict = trainer.update(data)
            loss_dict['cnt'] = 1
            add_dict(train_loss, loss_dict)

        cnt = train_loss.pop('cnt')
        log_loss_summary(train_loss, cnt, lambda x, y: log_string('Train {} is {}'.format(x, y)))

        if (epoch + 1) % cfg['freq']['save'] == 0:
            trainer.save()

        test_all()


if __name__ == '__main__':
    args = parse_args()
    torch.multiprocessing.set_start_method('spawn')
    main(args)

