import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation
import argparse
import torch

from create_arrow import create_direct_arrow

def vis_grasp(tip_pose, target_pose):
    if torch.is_tensor(tip_pose):
        tip_pose = tip_pose.cpu().detach().numpy().squeeze()
    if torch.is_tensor(target_pose):
        target_pose = target_pose.cpu().detach().numpy().squeeze()
    tips = []
    targets = []
    arrows = []
    color_code = np.array([[1,0,0],[0,1,0],[0,0,1],[1,1,0]])
    for i in range(4):
        tip = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
        tip.paint_uniform_color(color_code[i])
        tip.translate(tip_pose[i])
        target = o3d.geometry.TriangleMesh.create_sphere(radius=0.0025)
        target.paint_uniform_color(color_code[i] * 0.4)
        target.translate(target_pose[i])
        # create arrow point from tip to target
        arrow = create_direct_arrow(tip_pose[i], target_pose[i])
        arrow.paint_uniform_color(color_code[i])
        tips.append(tip)
        targets.append(target)
        arrows.append(arrow)
    return tips, targets, arrows

def vis_wrist_pose(
    pcd,
    pose,
    draw_frame=False,
    wrist_frame="springgrasp",
):
    geoms_list = [pcd]

    # Get wrist ref frame
    mesh_wrist = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.05
    )
    _wrist_R = Rotation.from_euler("XYZ",pose[3:])
    if wrist_frame == "springgrasp":
        wrist_R = (_wrist_R).as_matrix()
    else:
        # Transform from this wrist rotation to original local hand frame
        transform = Rotation.from_euler("xyz", [0, 90, 0], degrees=True)
        wrist_R = (_wrist_R*transform).as_matrix()
    mesh_wrist.translate(pose[:3])
    mesh_wrist.rotate(wrist_R)
    geoms_list.append(mesh_wrist)

    if draw_frame:
        # Global ref frame
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.1, origin=[0, 0, 0]
        )
        geoms_list.append(mesh_frame)

    o3d.visualization.draw_geometries(geoms_list)

def vis_results(
    pcd,
    init_ftip_pos,
    target_ftip_pos,
    draw_frame=False,
    wrist_pose=None,
    wrist_frame="springgrasp",
):
    # Get geometries to visualize grasp
    tips, targets, arrows = vis_grasp(init_ftip_pos, target_ftip_pos)

    geoms_list = [
        pcd, *tips, *targets, *arrows,
    ]

    # Draw reference frame
    if draw_frame:
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.1, origin=[0, 0, 0]
        )
        geoms_list.append(mesh_frame)
    
    # Draw wrist
    if wrist_pose is not None:
        mesh_wrist = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.05
        )
        wrist_pos = wrist_pose[:3]
        wrist_ori_XYZ = wrist_pose[3:]

        _wrist_R = Rotation.from_euler("XYZ",wrist_pose[3:])
        if wrist_frame == "springgrasp":
            wrist_R = (_wrist_R).as_matrix()
        else:
            # Transform from this wrist rotation to original local hand frame
            transform = Rotation.from_euler("xyz", [0, 90, 0], degrees=True)
            wrist_R = (_wrist_R*transform).as_matrix()

        mesh_wrist.translate(wrist_pos)
        mesh_wrist.rotate(wrist_R)
        geoms_list.append(mesh_wrist)

    o3d.visualization.draw_geometries(geoms_list)

def main(args):
    grasp_dict = np.load(args.grasp_path, allow_pickle=True)["data"].item() 
    grasp_i = 0

    pts = grasp_dict["input_pts"]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    init_ftip_pos = grasp_dict["start_tip_pose"][grasp_i]
    target_ftip_pos = grasp_dict["target_tip_pose"][grasp_i]
    palm_pose = grasp_dict["palm_pose"][grasp_i]

    vis_results(
        pcd,
        init_ftip_pos,
        target_ftip_pos,
        draw_frame=True,
        wrist_pose=palm_pose,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "grasp_path",
        type=str,
        help="Path to .npz file with grasp optimization results",
    )
    args = parser.parse_args()
    main(args)