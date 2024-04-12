"""
Get initial conditions for optimizer

wrist pose
joint angles
compliance
ftip target pos
"""
import numpy as np
import torch
from scipy.spatial.transform import Rotation
import itertools
import open3d as o3d

import viz_utils as v_utils

def get_init_wrist_pose_from_pcd(pcd, viz=False):
    """
    Get initial wrist poses

    returns:
        wrist_poses (np.array): [B, 6] array of B wrist poses
    """
    wrist_poses = []

    aabb = pcd.get_axis_aligned_bounding_box()
    center = aabb.get_center()
    extent = aabb.get_extent()

    # Fit points to plane and get normal
    pts = np.asarray(pcd.points)
    u, s, vt = np.linalg.svd(pts - center, full_matrices=False)
    normal = vt[-1] # Unit length

    normal_offset_list = [0.1]
    normal_dirs_list = [1, -1]
    base_x_rot_list = [0] # degrees
    base_y_rot_list = [0] # degrees
    base_z_rot_list = np.linspace(-180, 180, 25) # degrees

    xy_offset_list = np.linspace(0, max(extent)/2., num=3)[1:]
    xy_axes = [np.array([1, 0, 0]), np.array([0, 1, 0])]

    # TODO perturb pos randomly in all dimensions
    for params in itertools.product(
        normal_offset_list, normal_dirs_list,
        base_x_rot_list, base_y_rot_list, base_z_rot_list,
    ):
        offset = params[0]
        n_dir = params[1]
        x_rot = params[2]
        y_rot = params[3]
        z_rot = params[4]

        # Wrist pos is along plane normal
        base_pos = center + offset * normal * n_dir
        print(f"Base pose: normal offset {n_dir*offset} | rot ({x_rot}, {y_rot}, {z_rot})")
        
        # Define wrist ori so palm in perpendicular to plane normal
        # ie. in SpringGrasp convention, z axis is along plane normal
        R = np.zeros((3,3))
        # z-axis aligned with plane normal
        R[:, 2] = normal * n_dir

        # Rotate about x and y axis
        base_r = Rotation.from_matrix(R)
        xy_delta_r = Rotation.from_euler("xyz", [x_rot, y_rot, z_rot], degrees=True)
        # rotate base_r by xy_delta_r in its local current frame (post-multiplication)
        XYZ_ori = (base_r * xy_delta_r).as_euler("XYZ")

        # Create and append pose with base_pos (not translated along x and y) to list
        print(f"  Pose {len(wrist_poses)} xy offset from base pos: none")
        pose = np.concatenate((base_pos, XYZ_ori))
        wrist_poses.append(pose)
        # Visualize local wrist frame
        if viz:
            v_utils.vis_wrist_pose(pcd, pose, draw_frame=True)

        # Translate along x and y axes
        for trans_params in itertools.product(
            xy_axes, xy_offset_list, normal_dirs_list,
        ):
            axis = trans_params[0]
            xy_offset = trans_params[1]
            trans_dir = trans_params[2]
            H_robot_to_world = np.zeros((4,4))
            H_robot_to_world[:3, :3] = base_r.as_matrix()
            H_robot_to_world[:3, 3] = base_pos
            H_robot_to_world[3, 3] = 1
            pos_rf = xy_offset * axis * trans_dir # Pos in local frame
            pos = (H_robot_to_world @ np.append(pos_rf, 1))[:3]

            # Create and append pose to list
            print(f"  Pose {len(wrist_poses)} xy offset from base pos:", pos_rf)
            pose = np.concatenate((pos, XYZ_ori))
            wrist_poses.append(pose)

            # Visualize local wrist frame
            if viz:
                v_utils.vis_wrist_pose(pcd, pose, draw_frame=True)
    return np.array(wrist_poses)

def get_start_and_target_ftip_pos(wrist_poses, joint_angles, optimizer):
    """
    Get start and target fingertip positions for each 
    of the B wrist poses in wrist_poses

    args:
        wrist_poses (np.array): [B, 6] array of B poses. 
        joint_angles (Tensor): [B, 16] Tensor of B initial joint angles
        optimizer: optimizer with FK function

    returns:
        start_ftip_pos (Tensor): [B, 4, 3] of B start fingertip positions
        target_pose (Tensor): [B, 4, 3] of B target fingertip positions
    """
    start_ftip_pos = optimizer.forward_kinematics(
        joint_angles, torch.from_numpy(wrist_poses).cuda()
    )
    target_pose = start_ftip_pos.mean(dim=1, keepdim=True).repeat(1,4,1)
    target_pose = target_pose + (start_ftip_pos - target_pose) * 0.3
    return start_ftip_pos, target_pose