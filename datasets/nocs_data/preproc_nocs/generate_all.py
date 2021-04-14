import os
import sys
import argparse
from os.path import join as pjoin


def main(root_dset='../../../../data/nocs_data', data_type='all', parallel=True, num_proc=10):

    def execute(s):
        print(s)
        # os.system(s)

    syn_data = ['train', 'val']
    all_data = ['train', 'val', 'real_train', 'real_test']
    if data_type == 'test_only':
        syn_data = []
        all_data = ['real_test']

    categories = list(range(1, 7))

    root_dset = os.path.abspath(root_dset)

    ori_path = pjoin(root_dset, 'nocs_full')
    ikea_path = pjoin(root_dset, 'ikea_data')
    list_path = pjoin(root_dset, 'instance_list')
    model_path = pjoin(root_dset, 'model_corners')
    output_path = pjoin(root_dset, 'render')

    parallel_suffix = '' if not parallel else f' --parallel --num_proc={num_proc}'

    for data_type in syn_data:
        cmd = 'python match_table.py' + \
              f' --data_path={pjoin(ori_path, data_type)} --bg_path={ikea_path}' + \
              parallel_suffix
        execute(cmd)

    for data_type in all_data:
        cmd = 'python get_gt_poses.py' + \
              f' --data_path={ori_path} --data_type={data_type}' + \
              parallel_suffix
        execute(cmd)

    for data_type in all_data:
        cmd = 'python get_instance_list.py' + \
              f' --data_path={ori_path} --data_type={data_type} --list_path={list_path}' + \
              parallel_suffix
        execute(cmd)

    for data_type in all_data:
        for category in categories:
            cmd = 'python gather_instance_data.py' + \
                f' --data_path={ori_path} --data_type={data_type} --list_path={list_path}' + \
                f' --model_path={model_path} --output_path={output_path} --category={category}' + \
                parallel_suffix
            execute(cmd)

    if 'val' in all_data:
        execute(f'ln -s {pjoin(output_path, "val")} {pjoin(output_path, "test")}')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../../../../nocs_data')
    parser.add_argument('--data_type', type=str, default='all')
    parser.add_argument('--parallel', action='store_true')
    parser.add_argument('--num_proc', type=int, default=10)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    assert args.data_type in ['all', 'test_only']
    main(args.data_path, args.data_type, args.parallel, args.num_proc)