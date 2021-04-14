from os.path import join as pjoin

import numpy as np
import os
import sys
import pickle
from PIL import Image
from tqdm import tqdm

base_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(base_dir, '..'))

from data_utils import get_urdf_mobility, fetch_gt_bmvc, get_model_pts, farthest_point_sample

"""
root_dset/
    render/laptop/0/[0/1]/... (images)
    urdf/laptop/0/xxx.mobility
    # manually softlink the above
    preproc/laptop/0/[0/1]/...  # written by this file
"""


def point_cloud_from_depth(depth_image, camera_intrinsics):

    image_height = depth_image.shape[0]
    image_width = depth_image.shape[1]

    # Project depth into 3D pointcloud in camera coordinates
    pixel_x, pixel_y = np.meshgrid(np.linspace(0, image_width - 1, image_width),
                                   np.linspace(0, image_height - 1, image_height))
    pixel_y = image_height - 1 - pixel_y

    camera_points_x = np.multiply(pixel_x - camera_intrinsics[0, 2], depth_image / camera_intrinsics[0, 0])
    camera_points_y = np.multiply(pixel_y - camera_intrinsics[1, 2], depth_image / camera_intrinsics[1, 1])
    camera_points_z = -depth_image
    camera_points = np.array([camera_points_x, camera_points_y, camera_points_z]).transpose(1, 2, 0).reshape(-1,3)

    # Remove invalid 3D points (where depth == 0)
    valid_depth_ind = np.where(depth_image.flatten() > 0)[0]
    camera_points = camera_points[valid_depth_ind, :]

    return camera_points


def save_depth_pts_with_gt(original_path, output_path, obj_category, track_num, frame_num, num_parts,
                           model_pts_dict, urdf_dict, num_points=4096, is_debug=False):
    depth_file = pjoin(original_path, 'depth_filled', f'{frame_num:05d}.png')
    depth = Image.open(depth_file)
    depth = np.array(depth).astype(np.uint16) / 1000  # in mm to m
    part_masks = [None] * num_parts

    for p in range(num_parts):
        mask_file = pjoin(original_path, 'mask', f'{frame_num:05d}_00{p}.png')
        if not os.path.exists(mask_file):
            print('file does not exist', mask_file)
        mask_img = Image.open(mask_file)
        label = np.array(mask_img).astype(np.uint8)
        part_masks[p] = np.array((label[:, :] == 1)).astype(np.uint8)

    pts = model_pts_dict['pts']
    cpts = model_pts_dict['corners']

    parts_cam_pts = []
    parts_canon_pts = []

    parts_cam_cloud = []
    parts_canon_cloud = []

    info_path = pjoin(original_path, 'info')
    pose_dict, bbox_dict = fetch_gt_bmvc(info_path, frame_num, num_parts)

    # >>>>>>>>>> from canon to camera coordinates, to image coordinate
    for j in range(num_parts):
        parts_canon_pts.append(pts[j][0])
        part_canon_hom = np.concatenate((parts_canon_pts[j], np.ones((parts_canon_pts[j].shape[0], 1))), axis=1)
        part_cam = np.dot(part_canon_hom, pose_dict[j].T)[:, :3]

        perm = np.random.permutation(len(part_cam))
        parts_cam_pts.append(part_cam[perm[:num_points]])

    # >>>>>>>>>>> from image coordiante to camera coordinate, to canon
    for j in range(num_parts):
        cloud_cam = point_cloud_from_depth(depth * part_masks[j], camera_intrinsics)
        """
        idx = np.where(abs(cloud_cam[:, 2] - np.mean(cloud_cam[:, 2])) < thresh)[0]
        cloud_cam = cloud_cam[idx, :]
        """
        parts_cam_cloud.append(cloud_cam)

        cloud_cam_hom = np.concatenate((cloud_cam, np.ones((cloud_cam.shape[0], 1))), axis=1)
        camera_pose_mat = np.linalg.pinv(pose_dict[j].T)
        parts_canon_cloud.append(np.dot(cloud_cam_hom, camera_pose_mat)[:, :3])

    for k in range(num_parts):
        center_p = (cpts[k + 1][0] + cpts[k + 1][1]) / 2
        scale_xyz = cpts[k + 1][1] - cpts[k + 1][0]
        # print(center_p, scale_xyz)
        for d in range(3):
            if scale_xyz[d] > 0.05:
                idx = np.where(abs(parts_canon_cloud[k][:, d] - center_p[d]) < scale_xyz[d] / 2 + 0.005)[0]
            else:
                idx = np.where(abs(parts_canon_cloud[k][:, d] - center_p[d]) < scale_xyz[d] / 2 * 6)[0]
            parts_canon_cloud[k] = parts_canon_cloud[k][idx]
            parts_cam_cloud[k] = parts_cam_cloud[k][idx]
            if len(parts_cam_cloud[k]) < 3:
                print('data is bad', frame_num, k)
                # return None

    corners_list, factors_list = model_pts_dict['corners'], model_pts_dict['factors']
    corners_list = [np.array(corners) for corners in corners_list]

    parts_nocs_cloud = []

    def normalize(x, corner, factor):
        center = (corner[0] + corner[1]) * 0.5
        return (x - center) * factor

    for k in range(num_parts):
        parts_nocs_cloud.append(normalize(parts_canon_cloud[k], corners_list[k + 1], factors_list[k + 1]))

    nocs2camera = []

    for i in range(num_parts):
        p_scale = 1.0 / factors_list[i + 1]
        p_offset = np.mean(corners_list[i + 1], axis=0)
        p_trans = np.eye(4)
        p_trans[:3, 3] = p_offset

        urdf2camera = pose_dict[i]

        p_trans = urdf2camera @ p_trans

        p_rotation, p_translation = p_trans[:3, :3], p_trans[:3, 3:]

        nocs2camera.append({'scale': p_scale, 'rotation': p_rotation, 'translation': p_translation})

    all_pts, all_labels = [], []

    for j in range(num_parts):
        all_pts.append(parts_cam_cloud[j])
        all_labels.append(np.ones_like(parts_cam_cloud[j][:, 0]) * j)

    all_nocs = np.concatenate(parts_nocs_cloud, axis=0)

    all_pts = np.concatenate(all_pts, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    while len(all_pts) < num_points:
        all_pts = np.concatenate([all_pts, all_pts], axis=0)
        all_labels = np.concatenate([all_labels, all_labels], axis=0)
        all_nocs = np.concatenate([all_nocs, all_nocs], axis=0)

    idx = farthest_point_sample(all_pts, num_points)
    all_pts = all_pts[idx]
    all_labels = all_labels[idx]
    all_nocs = all_nocs[idx]

    output_path = pjoin(output_path, 'preproc', obj_category, '0', str(track_num), f'{frame_num:05d}.pkl')
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    with open(output_path, 'wb') as f:
        pickle.dump({'points': all_pts, 'labels': all_labels,
                     'nocs': all_nocs, 'nocs2camera': nocs2camera}, f)


if __name__=='__main__':
    output_path = '/orion/group/ArtPose/data/bmvc'
    input_path = '/orion/group/ArtPose/data/bmvc/BMVC15/test'
    # input_path = '../../bmvc_data/Laptop'
    # output_path = '../../bmvc_data'
    obj_category = 'laptop'
    camera_intrinsics = np.array([[540, 0, 323.65], [0, 540, 240.81], [0, 0, 1]])  # in pixels
    num_parts = 2
    urdf_dict = get_urdf_mobility(pjoin(output_path, 'urdf', obj_category, '0'))
    pts, norm_factors, corner_pts = get_model_pts(obj_file_list=urdf_dict['obj_name'])
    model_pts_dict = {'pts': pts, 'factors': norm_factors, 'corners': corner_pts}

    for track_num, track_name in enumerate(['Laptop_Seq_1', 'Laptop_Seq_2']):
        cur_input_path = pjoin(input_path, track_name)
        num_frames = len(os.listdir(pjoin(cur_input_path, 'depth_filled')))
        for frame_num in tqdm(range(num_frames)):
            save_depth_pts_with_gt(cur_input_path, output_path, obj_category, track_num, frame_num, num_parts,
                                   model_pts_dict, urdf_dict, num_points=4096, is_debug=False)
