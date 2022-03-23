import copy
import os
from typing import List
import random
import colorsys
import torch
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from geomdl.tessellate import make_triangle_mesh
from open3d import *
import open3d as o3d
from open3d import utility
from open3d import visualization
from transforms3d.affines import compose
from transforms3d.euler import euler2mat
from collections import Counter
import time

Vector3dVector, Vector3iVector = utility.Vector3dVector, utility.Vector3iVector


def get_np(data):
    if isinstance(data, torch.Tensor):
        return data.cpu().numpy()
    return data


def save_pcds_img():
    import matplotlib.pyplot as plt
    from transforms3d.affines import compose
    from transforms3d.euler import euler2mat

    get_test_data = get_test_dataset(config)
    R = euler2mat(-15 * 3.14 / 180, -35 * 3.14 / 180, 35)
    M = compose(T=(0, 0, 0), R=R, Z=(1, 1, 1))

    for test_id in range(config.num_test // config.batch_size):

        print("current val_id : {}".format(test_id))
        points_, labels, normals, primitives_, votes = next(get_test_data)[0]

        if test_id < 433:
            continue

        for b in range(config.batch_size):
            idx = test_id * config.batch_size + b
            point = points_[b].cpu().numpy() if isinstance(points_[b], torch.Tensor) else points_[b]
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(point)
            color = colorful_by_prim(primitives_[b])
            pcd.colors = o3d.utility.Vector3dVector(color)
            pcd.transform(M)

            vis = o3d.visualization.Visualizer()
            vis.create_window()
            vis.add_geometry(pcd)

            vis.poll_events()
            vis.update_renderer()
            time.sleep(2)

            # if idx == 0:
            #     vis.add_geometry(pcd)
            #     vis.run()
            # else:
            #     vis.add_geometry(pcd)
            #     vis.poll_events()
            #     vis.update_renderer()
            img_name = "/home/zhuhan/Code/ProjectMarch/parsenet-test/parsenet_img/{}.png".format(idx)
            vis.capture_screen_image(img_name, True)


def get_n_hls_colors(num):
    hls_colors = []
    i = 0
    step = 360.0 / num
    while i < 360:
        h = i
        s = 90 + random.random() * 10
        l = 50 + random.random() * 10
        _hlsc = [h / 360.0, l / 100.0, s / 100.0]
        hls_colors.append(_hlsc)
        i += step

    return hls_colors


def ncolors(num):
    rgb_colors = []
    if num < 1:
        return rgb_colors
    hls_colors = get_n_hls_colors(num)
    for hlsc in hls_colors:
        _r, _g, _b = colorsys.hls_to_rgb(hlsc[0], hlsc[1], hlsc[2])
        r, g, b = [x for x in (_r, _g, _b)]
        rgb_colors.append([r, g, b])

    return rgb_colors


def colorful_by_cluster(cluster_ids):
    if isinstance(cluster_ids, torch.Tensor):
        cluster_ids = cluster_ids.cpu().numpy()
    num_cluster = len(Counter(cluster_ids))
    color_map = {}
    new_color = ncolors(num_cluster)
    ret_col = np.zeros((len(cluster_ids), 3))
    for i in range(num_cluster):
        color_map[i] = new_color[i]
    for idx in range(len(cluster_ids)):
        if cluster_ids[idx] == -1:
            ret_col[idx] = [0.5, 0.5, 0.5]
        else:
            ret_col[idx] = color_map[cluster_ids[idx]]
    return ret_col


def colorful_by_prim(prims):
    prims = get_np(prims)
    assert max(prims) <= 9
    color_map = {  # tab-10
        1: [255, 127, 14],  # plane    orange
        3: [148, 103, 189],  # cone  purple
        4: [31, 119, 180],  # cyclinder  blue
        5: [44, 160, 44],  # sphere  green

        2: [227, 119, 194],  # open b-spline  pink
        8: [227, 119, 194],  # open b-spline  "extrusion"  pink

        0: [214, 39, 40],  # closed b-spline  red
        6: [214, 39, 40],  # closed b-spline  "other"  red
        7: [214, 39, 40],  # closed b-spline  "revolution"  red
        9: [214, 39, 40],  # closed b-spline  red
    }
    color = np.zeros((len(prims), 3))
    for idx, prim in enumerate(prims):
        if prim == -1:
            # color[idx] = [0.5, 0.5, 0.5]
            color[idx] = [1.0, 1.0, 1.0]
        else:
            color[idx] = [c / 255 for c in color_map[prim]]
    return Vector3dVector(color)


def vis_base_info(point, type, cluster, id=None, pred_cluster=[], pred_type=[]):
    """
    :param point: N*3
    :param type: N*1
    :param cluster: N*1
    """
    point, type, cluster = get_np(point), get_np(type), get_np(cluster)

    ori_pcd = o3d.geometry.PointCloud()
    type_pcd = o3d.geometry.PointCloud()
    cluster_pcd = o3d.geometry.PointCloud()

    ori_pcd.points = Vector3dVector(point)
    type_pcd.points = Vector3dVector(point + [1.5, 0, 0])
    cluster_pcd.points = Vector3dVector(point - [1.5, 0, 0])

    type_pcd.colors = colorful_by_prim(type)
    cluster_pcd.colors = Vector3dVector(colorful_by_cluster(cluster))
    window_name = "base info" if id == None else str(id)

    if len(pred_type) != 0 and len(pred_cluster) != 0:
        two_pcd = o3d.geometry.PointCloud()
        two_pcd.points = Vector3dVector(point - [0, 3, 0])
        two_pcd.colors = colorful_by_cluster(pred_type)
        ori_pcd.colors = colorful_by_cluster(pred_cluster)
        o3d.visualization.draw_geometries([ori_pcd, two_pcd, type_pcd, cluster_pcd], window_name=window_name)
    elif len(pred_cluster) != 0:
        ori_pcd.colors = colorful_by_cluster(pred_cluster)
        o3d.visualization.draw_geometries([ori_pcd, type_pcd, cluster_pcd], window_name=window_name)
    elif len(pred_type) != 0:
        ori_pcd.colors = colorful_by_prim(pred_type)
        o3d.visualization.draw_geometries([ori_pcd, type_pcd, cluster_pcd], window_name=window_name)
    else:
        o3d.visualization.draw_geometries([ori_pcd, type_pcd, cluster_pcd], window_name=window_name)
    return



def vis_named_points(points, cluster, required_cluster, if_vis_relation=False):
    points, cluster = get_np(points), get_np(cluster)
    pcd = o3d.geometry.PointCloud()
    pcd.points = Vector3dVector(points)
    colors = np.array([[0.74, 0.74, 0.74]] * len(points))

    id1, id2 = required_cluster
    require_idx = cluster == id1
    # colors[require_idx] = np.array([1., 0., 0.])
    colors[require_idx] = np.array([236 / 255, 116 / 255, 36 / 255])  # orange
    require_idx = cluster == id2
    # colors[require_idx] = np.array([0., 0., 1.])
    colors[require_idx] = np.array([152 / 255, 223 / 255, 138 / 255])
    # colors[cluster == 2] = np.array([0.12, 0.46, 0.70])
    # colors[cluster == 0] = np.array([152 / 255, 223 / 255, 138 / 255])
    # colors[cluster == 4] = np.array([0.95, 0.69, 0.51])
    pcd.colors = Vector3dVector(colors)
    if not if_vis_relation:
        o3d.visualization.draw_geometries([pcd], window_name=str(required_cluster))
        # o3d.io.write_point_cloud("/home/zhuhan/Code/ProjectMarch/a.ply", pcd)
    else:
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()
        time.sleep(0.2)

        img_name = "/home/zhuhan/Code/ProjectMarch/paper_img/relation_dect/159/{}.png".format(str(required_cluster))
        vis.capture_screen_image(img_name, True)
        o3d.io.write_point_cloud("/home/zhuhan/Code/ProjectMarch/paper_img/relation_dect/159/{}.ply".format(str(required_cluster)), pcd)



def vis_named_points_pairs(points, cluster, group_one, group_two, ):
    points, cluster = get_np(points), get_np(cluster)
    pcd = o3d.geometry.PointCloud()
    pcd.points = Vector3dVector(points)
    colors = np.array([[0.74, 0.74, 0.74]] * len(points))

    for group_one_idx in group_one:
        require_idx = cluster == group_one_idx
        colors[require_idx] = np.array([236 / 255, 116 / 255, 36 / 255])  # orange
    for group_two_idx in group_two:
        require_idx = cluster == group_two_idx
        colors[require_idx] = np.array([152 / 255, 223 / 255, 138 / 255])  # green
    pcd.colors = Vector3dVector(colors)

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()
    time.sleep(0.2)

    # img_name = "/home/zhuhan/Code/ProjectMarch/paper_img/relation_dect/369/{}.png".format(str(required_cluster))
    # img_name = "/home/zhuhan/Code/ProjectMarch/paper_img/relation_dect/369/pall.png")
    # vis.capture_screen_image(img_name, True)
    # o3d.io.write_point_cloud("/home/zhuhan/Code/ProjectMarch/paper_img/relation_dect/369/{}.ply".format(str(required_cluster)), pcd)
    o3d.io.write_point_cloud("/home/zhuhan/Code/ProjectMarch/paper_img/relation_dect/pall.ply", pcd)


def show_vote_result(inputs_xyz_sub, offset_per_point, V_gt, color_by, idx="don't know"):
    save_dir = "/home/zhuhan/Code/ProjectMarch/paper_img/9891"
    inputs_xyz_sub, offset_per_point, V_gt, I_gt = get_np(inputs_xyz_sub.permute(1,0)), get_np(offset_per_point), get_np(V_gt), get_np(color_by)
    colors = colorful_by_cluster(color_by)

    pcd = o3d.geometry.PointCloud()
    pcd.points = Vector3dVector(inputs_xyz_sub)
    pcd.colors = Vector3dVector(colors)

    vote_gt = o3d.geometry.PointCloud()
    V_gt[:, 0] -= 2.0
    vote_gt.points = Vector3dVector(inputs_xyz_sub+V_gt)
    vote_gt.colors = Vector3dVector(colors)

    vote_pred = o3d.geometry.PointCloud()
    offset_per_point[:,0] += 2.0
    vote_pred.points = Vector3dVector(inputs_xyz_sub+offset_per_point)
    vote_pred.colors = Vector3dVector(colors)

    all = [inputs_xyz_sub, inputs_xyz_sub + V_gt, inputs_xyz_sub + offset_per_point]
    all = np.concatenate(all)
    all_c = np.concatenate([colors, colors, colors])
    all_pcd = o3d.geometry.PointCloud()
    all_pcd.points = Vector3dVector(all)
    all_pcd.colors = Vector3dVector(all_c)
    o3d.io.write_point_cloud(os.path.join(save_dir, "vote_{}.ply".format(idx)), all_pcd)

    # o3d.visualization.draw_geometries([pcd, vote_gt, vote_pred], window_name=str(idx))

def take_photo():

    from tensorboard_logger import configure, log_value
    import matplotlib.pyplot as plt
    from transforms3d.affines import compose
    from transforms3d.euler import euler2mat

    R = euler2mat(-15 * 3.14 / 180, -35 * 3.14 / 180, 35)
    M = compose(T=(0, 0, 0), R=R, Z=(1, 1, 1))

    ply_dir = "/home/zhuhan/Code/ProjectMarch/paper_img/"
    ply_files = os.listdir(ply_dir)

    for one_ply in ply_files:
        if '.ply' not in one_ply:
            continue

        one_cloud = o3d.io.read_point_cloud(os.path.join(ply_dir, one_ply))
        point = np.asarray(one_cloud.points)
        color = np.asarray(one_cloud.colors)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point)
        pcd.colors = o3d.utility.Vector3dVector(color)
        # pcd.transform(M)
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(pcd)

        vis.poll_events()
        vis.update_renderer()
        time.sleep(0.2)

        img_name = "/home/zhuhan/Code/ProjectMarch/paper_img/photos/{}.png".format(str(one_ply.split(".")[0].split("_")[1]))
        vis.capture_screen_image(img_name, True)


def save_compare_result(point, gt, res1, res2, index):
    save_dir = "/home/zhuhan/Code/ProjectMarch/paper_img/compare_new/"
    point, gt, res1, res2 = get_np(point), get_np(gt), get_np(res1), get_np(res2)
    gt_pcd = o3d.geometry.PointCloud()
    res1_pcd = o3d.geometry.PointCloud()
    res2_pcd = o3d.geometry.PointCloud()
    gt_pcd.points = Vector3dVector(point)
    res1_pcd.points = Vector3dVector(point-[3, 0, 0])
    res2_pcd.points = Vector3dVector(point+[1.5, 0, 0])

    gt_pcd.colors = Vector3dVector(colorful_by_cluster(gt))
    res1_pcd.colors = Vector3dVector(colorful_by_cluster(res1))
    res2_pcd.colors = Vector3dVector(colorful_by_cluster(res2))

    all = [point, point-[3, 0, 0], point+[1.5, 0, 0]]
    all = np.concatenate(all)
    all_c = np.concatenate([colorful_by_cluster(gt), colorful_by_cluster(res1), colorful_by_cluster(res2)])
    all_pcd = o3d.geometry.PointCloud()
    all_pcd.points = Vector3dVector(all)
    all_pcd.colors = Vector3dVector(all_c)

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(all_pcd)
    vis.poll_events()
    vis.update_renderer()
    time.sleep(0.2)
    img_name = "/home/zhuhan/Code/ProjectMarch/paper_img/compare_new/{}.png".format(str(index))
    vis.capture_screen_image(img_name, True)
    o3d.io.write_point_cloud(os.path.join(save_dir, "compare_{}.ply".format(index)), all_pcd)


if __name__ == '__main__':
    take_photo()
