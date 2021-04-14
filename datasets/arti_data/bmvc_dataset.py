import os
import sys
from os.path import join as pjoin
import pickle

base_dir = os.path.dirname(__file__)
sys.path.append(base_dir)
sys.path.append(os.path.join(base_dir, '..'))

from arti_data_process import generate_instance_info


class BMVCDataset:
    def __init__(self, root_dset, obj_category, track=0, truncate_length=None):
        super(BMVCDataset, self).__init__()
        self.data_path = pjoin(root_dset, 'preproc', obj_category, '0', str(track))
        self.len = len(os.listdir(self.data_path))
        if truncate_length is not None:
            self.len = min(self.len, truncate_length)
        instance = '0'
        self.model_info_dict = {instance: generate_instance_info(root_dset, obj_category, instance)}
        model_info = self.model_info_dict[instance]
        self.ins_info = {instance: {'corners': [model_info['global_corner']] + model_info['corner']}}

    def __getitem__(self, i):
        data_path = pjoin(self.data_path, f'{i:05d}.pkl')
        with open(data_path, 'rb') as f:
            full_data = pickle.load(f)

        nocs2camera = full_data.pop('nocs2camera')
        return {'data': full_data,
                'meta': {'path': data_path,
                         'nocs2camera': nocs2camera}
                }

    def __len__(self):
        return self.len




