import numpy as np
import pickle
import os.path
import trimesh
import glob
import torch
import os
import sys
from os.path import join as pjoin
BASEPATH = os.path.dirname(__file__)
sys.path.insert(0, BASEPATH)
sys.path.insert(0, pjoin(BASEPATH, '..'))

import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
plt.ioff()

from transformations import euler_matrix, quaternion_matrix
from misc.visualize.vis_utils import plot3d_pts
from network.models.pointnet_utils import farthest_point_sample as farthest_point_sample_cuda
from utils import ensure_dirs

import cv2

epsilon = 1e-8


def split_dataset(root_dset, obj_category, num_expr, test_ins, temporal=False):
    """
    write train.txt, val.txt, test.txt
    """

    path = pjoin(root_dset, "render", obj_category)
    train_ins = [ins for ins in os.listdir(path) if ins not in test_ins and not ins.startswith('.')]
    print(train_ins)
    print(test_ins)
    print(f'len train_ins = {len(train_ins)}, len test_ins = {len(test_ins)}')

    train_list, val_list, test_list = [], [], []
    for instance in train_ins:
        all_tracks = []
        for track_dir in glob.glob(pjoin(path, instance, '*')):
            frames = glob.glob(pjoin(track_dir, 'cloud', '*'))
            cloud_list = [file for file in frames if file.endswith('.npz')]
            cloud_list.sort(key=lambda str: int(str.split('.')[-2].split('/')[-1]))
            if not temporal:  # pick a random view point for each art
                train_list += cloud_list[:-1]
                val_list += cloud_list[-1:]
            else:
                all_tracks.append(cloud_list)
        if temporal:
            train_list += [item for sublist in all_tracks[:-2] for item in sublist]
            val_list += [item for sublist in all_tracks[-2:] for item in sublist]

    for instance in test_ins:
        for track_dir in glob.glob(pjoin(path, instance, '*')):
            frames = glob.glob(pjoin(track_dir, 'cloud', '*'))
            cloud_list = [file for file in frames if file.endswith('.npz')]
            cloud_list.sort(key=lambda str: int(str.split('.')[-2].split('/')[-1]))
            test_list += cloud_list

    output_path = pjoin(root_dset, "splits", obj_category, num_expr)
    ensure_dirs(output_path)
    name_list = ['train.txt', 'val.txt', 'test.txt']
    for name, content in zip(name_list, [train_list, val_list, test_list]):
        print(f'{name}: {len(content)}')
        with open(pjoin(output_path, name), 'w') as f:
            for item in content:
                f.write('{}\n'.format(item))


def split_seq_dataset(root_dset, obj_category, num_expr, test_ins):
    """
    write train_seq.txt, test_seq.txt
    """

    path = pjoin(root_dset, "render_seq", obj_category)
    train_ins = [ins for ins in os.listdir(path) if ins not in test_ins and not ins.startswith('.')]
    print(train_ins)
    print(test_ins)
    print(f'len train_ins = {len(train_ins)}, len test_ins = {len(test_ins)}')

    train_list, test_list = [], []
    for data_list, ins_list in zip([train_list, test_list], [train_ins, test_ins]):
        for instance in ins_list:
            for track_dir in glob.glob(pjoin(path, instance, '*')):
                frames = glob.glob(pjoin(track_dir, 'cloud', '*'))
                cloud_list = [file for file in frames if file.endswith('.npz')]
                cloud_list.sort(key=lambda str: int(str.split('.')[-2].split('/')[-1]))
                data_list += cloud_list

    output_path = pjoin(root_dset, "splits", obj_category, num_expr)
    ensure_dirs(output_path)

    name_list = ['train_seq.txt', 'test_seq.txt']
    for name, content in zip(name_list, [train_list, test_list]):
        print(f'{name}: {len(content)}')
        with open(pjoin(output_path, name), 'w') as f:
            for item in content:
                f.write('{}\n'.format(item))


def split_real_dataset(root_dset, obj_category, num_expr, test_ins):
    """
    write real_test.txt
    """

    path = pjoin(root_dset, "render", obj_category)

    test_list = []
    for instance in test_ins:
        for track_dir in glob.glob(pjoin(path, instance, '*')):
            frames = glob.glob(pjoin(track_dir, 'cloud', '*'))
            cloud_list = [file for file in frames if file.endswith('.npz')]
            cloud_list.sort(key=lambda str: int(str.split('.')[-2].split('/')[-1]))
            test_list += cloud_list

    output_path = pjoin(root_dset, "splits", obj_category, num_expr)
    ensure_dirs(output_path)
    with open(pjoin(output_path, 'real_test.txt'), 'w') as f:
        for item in test_list:
            f.write('{}\n'.format(item))


def get_seq_file_list_index(file_list):
    track_dict = {}
    start_points = []
    for i, file_name in enumerate(file_list):
        track_name = '/'.join(file_name.split('/')[:-1])
        if track_name not in track_dict:
            track_dict[track_name] = i
            start_points.append(i)
    print(track_dict)
    start_points.append(len(file_list))
    return start_points


def farthest_point_sample(xyz, npoint, device):
    """
    Input:
        xyz: pointcloud data, [N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint]
    """
    if torch.cuda.is_available():
        if len(xyz) > 5 * npoint:
            idx = np.random.permutation(len(xyz))[:5 * npoint]
            torch_xyz = torch.tensor(xyz[idx]).float().to(device).reshape(1, -1, 3)
            torch_idx = farthest_point_sample_cuda(torch_xyz, npoint)
            torch_idx = torch_idx.cpu().numpy().reshape(-1)
            idx = idx[torch_idx]
            return idx
        else:
            torch_xyz = torch.tensor(xyz).float().to(device).reshape(1, -1, 3)
            torch_idx = farthest_point_sample_cuda(torch_xyz, npoint)
            idx = torch_idx.reshape(-1).cpu().numpy()
            return idx
    else:
        print('FPS on CPU: use random sampling instead')
        idx = np.random.permutation(len(xyz))[:npoint]
        return idx

    N, C = xyz.shape
    centroids = np.zeros((npoint, ), dtype=int)
    distance = np.ones((N, )) * 1e10
    farthest = np.random.randint(0, N, dtype=int)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :].reshape(1, C)
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    return centroids


def random_point_sample(xyz, npoint):
    return np.random.permutation(len(xyz))[:npoint]


def get_obj2link_dict(urdf_dict):
    """
        from .obj to link frame
        Note the "k + 1" -- it's in accordance w/ get_mobility_urdf
    """
    num_parts = urdf_dict['num_links'] - 1  # exclude the base
    obj2link_dict = {}
    for k in range(num_parts):
        parts_urdf_pos = np.array(urdf_dict['link']['xyz'][k + 1])
        parts_urdf_pos = np.reshape(parts_urdf_pos, (-1))
        parts_urdf_orn = np.array(urdf_dict['link']['rpy'][k + 1])
        parts_urdf_orn = np.reshape(parts_urdf_orn, (-1))
        center_joint_orn = parts_urdf_orn
        obj2link_r = euler_matrix(center_joint_orn[0],
                                  center_joint_orn[1],
                                  center_joint_orn[2])[:4, :4]
        obj2link_t = parts_urdf_pos
        obj2urdf_mat = obj2link_r
        obj2urdf_mat[:3, 3] = obj2link_t[:3]
        obj2link_dict[k] = obj2urdf_mat

    return obj2link_dict


def pose_pq_to_mat(pq):  # pq: tuple((3), (4))
    mat = quaternion_matrix(pq[1])
    mat[:3, 3] = pq[0]
    return mat


def read_gt_pose_dict(gt_dict):
    cam2world_pose = pose_pq_to_mat(gt_dict['camera_pose'])
    link2world_dict = {key: pose_pq_to_mat(pq) for key, pq in gt_dict['link_pose'].items()}
    return cam2world_pose, link2world_dict


def multiply_pose(pose_a, pose_b):  # a(b())
    keys_a = None if not isinstance(pose_a, dict) else list(pose_a.keys())
    keys_b = None if not isinstance(pose_b, dict) else list(pose_b.keys())
    keys = keys_b if keys_a is None else keys_a
    if keys is None:
        return np.matmul(pose_a, pose_b)
    else:
        result = {}
        for key in keys:
            a = pose_a if keys_a is None else pose_a[key]
            b = pose_b if keys_b is None else pose_b[key]
            result[key] = np.matmul(a, b)
        return result


def inv_pose(pose):
    if isinstance(pose, dict):
        return {key: np.linalg.inv(value) for key, value in pose.items()}
    else:
        return np.linalg.inv(pose)


def pose2srt(pose):
    if isinstance(pose, dict):
        num_parts = len(pose)
        return [pose2srt(pose[p]) for p in range(num_parts)]
    else:
        scale = 1.0 / pose[3, 3]
        rotation = pose[:3, :3]
        trans = pose[:3, 3:] * scale
        return {'rotation': rotation, 'translation': trans, 'scale': scale}


def get_obj2norm_pose(corner, factor):  # (obj - corner / 2) * factor = obj * factor - center * factor
    scaling = np.eye(4)
    scaling[3, 3] = 1.0 / factor
    center = (corner[0] + corner[1]) * 0.5
    trans = np.eye(4)
    trans[:3, 3] = -center * factor
    obj2norm = np.matmul(trans, scaling)
    return obj2norm


def get_urdf_mobility(inpath, verbose=False):
    if not inpath.endswith(".urdf"):
        urdf_name = inpath + "/mobility.urdf"
    else:
        urdf_name = inpath
        inpath = '/'.join(inpath.split('/')[:-1])

    urdf_ins = {}
    tree_urdf = ET.parse(urdf_name)
    num_real_links = len(tree_urdf.findall('link'))
    root_urdf = tree_urdf.getroot()
    rpy_xyz = {}
    list_xyz = [None] * num_real_links
    list_rpy = [None] * num_real_links
    list_obj = [None] * num_real_links
    # ['obj'] ['link/joint']['xyz/rpy'] [0, 1, 2, 3, 4]
    num_links = 0
    for link in root_urdf.iter('link'):
        num_links += 1
        if link.attrib['name'] == 'base':
            index_link = 0
        else:
            index_link = int(link.attrib['name'].split('_')[1]) + 1  # since the name is base, link_0, link_1
        list_xyz[index_link] = []
        list_rpy[index_link] = []
        list_obj[index_link] = []
        for visual in link.iter('visual'):
            for origin in visual.iter('origin'):
                if 'xyz' in origin.attrib:
                    list_xyz[index_link].append([float(x) for x in origin.attrib['xyz'].split()])
                else:
                    list_xyz[index_link].append([0, 0, 0])
                if 'rpy' in origin.attrib:
                    list_rpy[index_link].append([float(x) for x in origin.attrib['rpy'].split()])
                else:
                    list_rpy[index_link].append([0, 0, 0])
            for geometry in visual.iter('geometry'):
                for mesh in geometry.iter('mesh'):
                    if 'home' in mesh.attrib['filename'] or 'work' in mesh.attrib['filename']:
                        list_obj[index_link].append(mesh.attrib['filename'])
                    else:
                        list_obj[index_link].append(inpath + '/' + mesh.attrib['filename'])

    rpy_xyz['xyz'] = list_xyz
    rpy_xyz['rpy'] = list_rpy  # here it is empty list
    urdf_ins['link'] = rpy_xyz
    urdf_ins['obj_name'] = list_obj

    rpy_xyz = {}
    list_type = [None] * (num_real_links - 1)
    list_parent = [None] * (num_real_links - 1)
    list_child = [None] * (num_real_links - 1)
    list_xyz = [None] * (num_real_links - 1)
    list_rpy = [None] * (num_real_links - 1)
    list_axis = [None] * (num_real_links - 1)
    list_limit = [[0, 0]] * (num_real_links - 1)
    # here we still have to read the URDF file
    for joint in root_urdf.iter('joint'):
        """
        joint_index = int(joint.attrib['name'].split('_')[1])
        list_type[joint_index] = joint.attrib['type']
        """

        for child in joint.iter('child'):
            link_name = child.attrib['link']
            if link_name == 'base':
                link_index = 0
            else:
                link_index = int(link_name.split('_')[1]) + 1
            joint_index = link_index - 1  # !!! the joint's index should be the same as its child's index - 1
            list_child[joint_index] = link_index

        list_type[joint_index] = joint.attrib['type']

        for parent in joint.iter('parent'):
            link_name = parent.attrib['link']
            if link_name == 'base':
                link_index = 0
            else:
                link_index = int(link_name.split('_')[1]) + 1
            list_parent[joint_index] = link_index

        for origin in joint.iter('origin'):
            if 'xyz' in origin.attrib:
                list_xyz[joint_index] = [float(x) for x in origin.attrib['xyz'].split()]
            else:
                list_xyz[joint_index] = [0, 0, 0]
            if 'rpy' in origin.attrib:
                list_rpy[joint_index] = [float(x) for x in origin.attrib['rpy'].split()]
            else:
                list_rpy[joint_index] = [0, 0, 0]
        for axis in joint.iter('axis'):  # we must have
            list_axis[joint_index] = [float(x) for x in axis.attrib['xyz'].split()]
        for limit in joint.iter('limit'):
            list_limit[joint_index] = [float(limit.attrib['lower']), float(limit.attrib['upper'])]

    rpy_xyz['type'] = list_type
    rpy_xyz['parent'] = list_parent
    rpy_xyz['child'] = list_child
    rpy_xyz['xyz'] = list_xyz
    rpy_xyz['rpy'] = list_rpy
    rpy_xyz['axis'] = list_axis
    rpy_xyz['limit'] = list_limit

    urdf_ins['joint'] = rpy_xyz
    urdf_ins['num_links'] = num_real_links
    if verbose:
        for j, pos in enumerate(urdf_ins['link']['xyz']):
            if len(pos) > 3:
                print('link {} xyz: '.format(j), pos[0])
            else:
                print('link {} xyz: '.format(j), pos)
        for j, orient in enumerate(urdf_ins['link']['rpy']):
            if len(orient) > 3:
                print('link {} rpy: '.format(j), orient[0])
            else:
                print('link {} rpy: '.format(j), orient)
        # for joint
        for j, pos in enumerate(urdf_ins['joint']['xyz']):
            print('joint {} xyz: '.format(j), pos)
        for j, orient in enumerate(urdf_ins['joint']['rpy']):
            print('joint {} rpy: '.format(j), orient)
        for j, orient in enumerate(urdf_ins['joint']['axis']):
            print('joint {} axis: '.format(j), orient)
        for j, child in enumerate(urdf_ins['joint']['child']):
            print('joint {} has child link: '.format(j), child)
        for j, parent in enumerate(urdf_ins['joint']['parent']):
            print('joint {} has parent link: '.format(j), parent)

    return urdf_ins


def get_all_objs(obj_file_list, is_debug=False, verbose=False):
    pts_list = []  # for each link, a list of vertices
    name_list = []  # for each link, the .obj filenames
    offset = 0

    def read_obj_file(obj_file):
        try:
            tm = trimesh.load(obj_file)
            vertices_obj = np.array(tm.vertices)
        except:
            dict_mesh, _, _, _ = load_model_split(obj_file)
            vertices_obj = np.concatenate(dict_mesh['v'], axis=0)
        return vertices_obj

    for k, obj_files in enumerate(obj_file_list):  # each corresponds to a link
        cur_list = None
        if isinstance(obj_files, list):
            cur_list = obj_files
        elif obj_files is not None:
            cur_list = [obj_files]
        if verbose:
            print('{} part has {} obj files'.format(k, len(cur_list)))
        # collect all names & vertices
        part_pts = []
        name_objs = []
        for obj_file in cur_list:
            if obj_file is not None and not isinstance(obj_file, list):
                vertices_obj = read_obj_file(obj_file)
                part_pts.append(vertices_obj)
                name_obj = obj_file.split('.')[0].split('/')[-1]
                name_objs.append(name_obj)

        part_pts_all = np.concatenate(part_pts, axis=0)
        pts_list.append(part_pts_all + offset)
        name_list.append(name_objs)  # which should follow the right

    if is_debug:
        print('name_list is: ', name_list)

    # vertices: a list of sublists,
    # sublists contain vertices in the whole shape (0) and in each part (1, 2, ..)
    vertices = [pts_list]
    for part in pts_list:
        vertices.append([part])

    norm_factors = []  # for each link, a float
    corner_pts = []
    # calculate bbox & corners for the whole object
    # as well as each part
    for j in range(len(vertices)):

        part_verts = np.concatenate(vertices[j], axis=0)  # merge sublists
        pmax, pmin = np.amax(part_verts, axis=0), np.amin(part_verts, axis=0)
        corner_pts.append([pmin, pmax])  # [index][left/right][x, y, z], numpy array
        norm_factor = np.sqrt(1) / np.sqrt(np.sum((pmax - pmin) ** 2))
        norm_factors.append(norm_factor)
        if verbose:
            plot3d_pts([[part_verts[::2]]], ['model pts'], s=15, title_name=['GT model pts {}'.format(j)],
                       sub_name=str(j))

    return vertices[1:], norm_factors, corner_pts


def get_model_pts(obj_file_list, is_debug=False):
    """
    For item obj_category/item,
    get_urdf(_mobility) returns obj_file_list:
        [[objs of 0th link], [objs of 1st link], ...]
    This function reads these obj files,
    and calculates the following for each link (part) --- except the first link (base link)
    - model parts,
    - norm_factors, 1 / diagonal of bbox
    - corner_pts, the corners of the bbox

    """
    if obj_file_list is not None and obj_file_list[0] == []:
        if is_debug:
            print('removing the first obj list, which corresponds to base')
        obj_file_list = obj_file_list[1:]

    model_pts, norm_factors, corner_pts = get_all_objs(obj_file_list=obj_file_list,
                                                       is_debug=is_debug)
    return model_pts, norm_factors, corner_pts


def part_canon2urdf_from_urdf_dict(urdf_dict):
    """
        part_canon2urdf: from link frame to urdf frame in rest pose
        Assumption: ".obj"s are in the rest pose,
            which means, urdf_dict['link']['xyz'] + link position in rest pose = 0
            (.obj -> put at 'link''xyz' w.r.t. link frame;
             link frame -> put at link position in urdf frame (rest pose))

        Note the "k + 1" -- it's in accordance w/ get_mobility_urdf
    """
    num_parts = urdf_dict['num_links'] - 1 # exclude the base
    parts_canon2urdf = [None] * num_parts
    for k in range(num_parts):
        parts_urdf_pos = -np.array(urdf_dict['link']['xyz'][k + 1])
        parts_urdf_pos = np.reshape(parts_urdf_pos, (-1))

        parts_urdf_orn = np.array(urdf_dict['link']['rpy'][k + 1])
        parts_urdf_orn = np.reshape(parts_urdf_orn, (-1))

        center_joint_orn = parts_urdf_orn
        my_canon2urdf_r = euler_matrix(center_joint_orn[0],
                                       center_joint_orn[1],
                                       center_joint_orn[2])[:4, :4]
        my_canon2urdf_t = parts_urdf_pos
        my_canon2urdf_mat = my_canon2urdf_r
        my_canon2urdf_mat[:3, 3] = my_canon2urdf_t[:3]
        parts_canon2urdf[k] = my_canon2urdf_mat

    return parts_canon2urdf


def fetch_gt_bmvc(info_path, frame_num, num_parts):
    pose_dict = {}
    bbox_dict = {}

    for k in range(num_parts):
        info_file = pjoin(info_path, f'info_{frame_num:05d}_{k:03d}.txt')
        with open(info_file, "r", errors='replace') as fp:
            line = fp.readline()
            viewMat = np.eye(4) # from object coordinate to camera coordinate
            tight_bb = np.zeros((3))
            while line:
                if len(line.strip()) == 9 and line.strip()[:8] == 'rotation':
                    for i in range(3):
                        line = fp.readline()
                        viewMat[i, :3] = [float(x) for x in line.strip().split()]
                if len(line.strip()) == 7 and line.strip()[:6] == 'center':
                    line = fp.readline()
                    viewMat[:3, 3] = [float(x) for x in line.strip().split()]
                if len(line.strip()) == 7 and line.strip()[:6] == 'extent':
                    line = fp.readline()
                    tight_bb[:] = [float(x) for x in line.strip().split()]
                    break
                line = fp.readline()
        pose_dict[k] = viewMat
        bbox_dict[k] = tight_bb
    return pose_dict, bbox_dict


def load_model_split(inpath):
    vsplit = []
    fsplit = []
    dict_mesh = {}
    list_group = []
    list_xyz = []
    list_face = []
    with open(inpath, "r", errors='replace') as fp:
        line = fp.readline()
        cnt = 1
        while line:
            if len(line) < 2:
                line = fp.readline()
                cnt += 1
                continue
            xyz = []
            face = []
            if line[0] == 'g':
                list_group.append(line[2:])
            if line[0:2] == 'v ':
                vcount = 0
                while line[0:2] == 'v ':
                    xyz.append([float(coord) for coord in line[2:].strip().split()])
                    vcount += 1
                    line = fp.readline()
                    cnt += 1
                vsplit.append(vcount)
                list_xyz.append(xyz)

            if line[0] == 'f':
                fcount = 0
                while line[0] == 'f':
                    face.append([num for num in line[2:].strip().split()])
                    fcount += 1
                    line = fp.readline()
                    cnt += 1
                    if not line:
                        break
                fsplit.append(fcount)
                list_face.append(face)
            line = fp.readline()
            cnt += 1
    dict_mesh['v'] = list_xyz
    dict_mesh['f'] = list_face

    return dict_mesh, list_group, vsplit, fsplit


def cam_to_image(camera_matrix, cam_points):
    permutation = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
    img_points = np.matmul(camera_matrix,
                           np.matmul(permutation.T, cam_points.T)).T  # [N, 3]
    img_points[..., :2] /= img_points[..., 2:]
    return img_points[..., :2]


def get_pts_for_bbox_overlay(root_dset, obj_category, instance, track_num, frame_i, num_parts, num_points=4096,
                             synthetic=False, suffix=''):
    from datasets.arti_data.arti_data_process import read_cloud

    preproc_name = f'preproc{suffix}'
    render_name = f'render{suffix}'
    cloud_path = pjoin(root_dset, render_name, obj_category, instance, f'{track_num}', 'cloud', f'{frame_i}.npz')
    preproc_path = pjoin(root_dset, preproc_name, obj_category, instance, f'{track_num}', 'cloud', f'{frame_i}.pkl')
    if os.path.exists(preproc_path):
        with open(preproc_path, 'rb') as f:
            preproc_dict = pickle.load(f)
            cam_points, seg = preproc_dict['cam'], preproc_dict['seg']
    else:
        cloud_dict = np.load(cloud_path, allow_pickle=True)
        if 'all_dict' in cloud_dict:
            cloud_dict = cloud_dict['all_dict'].item()
            cam_points, seg = read_cloud(cloud_dict, num_points, synthetic=synthetic,
                                         num_parts=num_parts if not synthetic else None)
        else:
            cam_points = cloud_dict['point']
            return [cam_points]

    pt_list = []
    for i in range(max(seg) + 1):
        idx = np.where(seg == i)[0]
        pt_list.append(cam_points[idx])

    return pt_list
