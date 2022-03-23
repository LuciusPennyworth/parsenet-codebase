"""
This script defines dataset loading for the segmentation task on ABC dataset.
"""
import os.path

import h5py
import numpy as np

from src.augment_utils import rotate_perturbation_point_cloud, jitter_point_cloud, shift_point_cloud, \
    random_scale_point_cloud, rotate_point_cloud

EPS = np.finfo(np.float32).eps


class Dataset:
    def __init__(self,
                 batch_size,
                 train_size=None,
                 val_size=None,
                 test_size=None,
                 normals=False,
                 primitives=False,
                 if_train_data=True,
                 prefix=""):
        """
        Dataset of point cloud from ABC dataset.
        :param root_path:
        :param batch_size:
        :param if_train_data: since training dataset is large and consumes RAM,
        we can optionally choose to not load it.
        """
        self.batch_size = batch_size
        self.normals = normals
        self.primitives = primitives
        self.augment_routines = [rotate_perturbation_point_cloud, jitter_point_cloud, shift_point_cloud,
                                 random_scale_point_cloud, rotate_point_cloud]

        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.prefix = prefix

    def get_train(self, randomize=False, augment=False, anisotropic=False, align_canonical=False,
                  if_normal_noise=False):
        train_size = self.train_size

        while (True):
            l = np.arange(train_size)
            if randomize:
                np.random.shuffle(l)
            all_index = l

            for i in range(train_size // self.batch_size):
                current_batch_idx = all_index[i * self.batch_size:(i + 1) * self.batch_size]
                all_points, all_normals, = [], []
                all_labels, all_primitives, all_parameters = [], [], []

                for idx in range(self.batch_size):
                    data_file = os.path.join(self.prefix, "{:05d}.h5".format(current_batch_idx[idx]))
                    with h5py.File(data_file, 'r') as hf:
                        points = np.array(hf.get("points"))
                        normals = np.array(hf.get("normals"))
                        labels = np.array(hf.get("labels"))
                        primitives = np.array(hf.get("prim"))
                        primitive_param = np.array(hf.get("T_param"))

                    all_points.append(points)
                    all_normals.append(normals)
                    all_labels.append(labels)
                    all_primitives.append(primitives)
                    all_parameters.append(primitive_param)

                all_points = np.stack(all_points, 0)
                all_normals = np.stack(all_normals, 0)
                all_labels = np.stack(all_labels, 0)
                all_primitives = np.stack(all_primitives, 0)
                all_parameters = np.stack(all_parameters, 0)

                if augment:
                    c = np.random.choice(np.arange(5))
                    all_points = self.augment_routines[c](all_points)
                if if_normal_noise:
                    noise = all_normals * np.clip(np.random.randn(1, all_points.shape[1], 1) * 0.01, a_min=-0.01, a_max=0.01)
                    all_points = all_points + noise.astype(np.float32)

                # VOTE
                B, N, _ = all_points.shape
                all_cluster_vote = np.zeros((B, N, 3))
                for i in range(B):
                    one_unique_cluster = np.unique(all_labels[i])
                    one_point = all_points[i]
                    one_label = all_labels[i]
                    for c in one_unique_cluster:
                        cluster_idx = np.where(one_label == c)
                        cluster_point = np.array(one_point[cluster_idx])
                        center = 0.5 * (cluster_point.min(0) + cluster_point.max(0))
                        all_cluster_vote[i, cluster_idx, :] = center - cluster_point

                return_items = [all_points, all_labels, all_normals, all_primitives, all_parameters, all_cluster_vote, current_batch_idx]

                yield return_items

    def get_test(self, randomize=False, anisotropic=False, align_canonical=False, if_normal_noise=False):
        test_size = self.test_size
        batch_size = self.batch_size
        all_index = np.arange(test_size) + 50000
        while (True):

            for i in range(test_size // batch_size):
                current_batch_idx = all_index[i * self.batch_size:(i + 1) * self.batch_size]
                all_points, all_normals, = [], []
                all_labels, all_primitives, all_parameters = [], [], []

                for idx in range(self.batch_size):
                    data_file = os.path.join(self.prefix, "{:05d}.h5".format(current_batch_idx[idx]))
                    with h5py.File(data_file, 'r') as hf:
                        points = np.array(hf.get("points"))
                        normals = np.array(hf.get("normals"))
                        labels = np.array(hf.get("labels"))
                        primitives = np.array(hf.get("prim"))
                        primitive_param = np.array(hf.get("T_param"))

                    all_points.append(points)
                    all_normals.append(normals)
                    all_labels.append(labels)
                    all_primitives.append(primitives)
                    all_parameters.append(primitive_param)

                all_points = np.stack(all_points, 0)
                all_normals = np.stack(all_normals, 0)
                all_labels = np.stack(all_labels, 0)
                all_primitives = np.stack(all_primitives, 0)
                all_parameters = np.stack(all_parameters, 0)

                if if_normal_noise:
                    noise = all_normals * np.clip(np.random.randn(1, all_points.shape[1], 1) * 0.01, a_min=-0.01, a_max=0.01)
                    all_points = all_points + noise.astype(np.float32)

                # VOTE
                B, N, _ = all_points.shape
                all_cluster_vote = np.zeros((B, N, 3))
                for i in range(B):
                    one_unique_cluster = np.unique(all_labels[i])
                    one_point = all_points[i]
                    one_label = all_labels[i]
                    for c in one_unique_cluster:
                        cluster_idx = np.where(one_label == c)
                        cluster_point = np.array(one_point[cluster_idx])
                        center = 0.5 * (cluster_point.min(0) + cluster_point.max(0))
                        all_cluster_vote[i, cluster_idx, :] = center - cluster_point

                return_items = [all_points, all_labels, all_normals, all_primitives, all_parameters, all_cluster_vote, current_batch_idx]

                yield return_items

    def get_val(self, randomize=False, anisotropic=False, align_canonical=False, if_normal_noise=False):
        val_size = self.val_size
        batch_size = self.batch_size
        all_index = np.arange(val_size) + 40000

        while (True):
            for i in range(val_size // batch_size):
                current_batch_idx = all_index[i * self.batch_size:(i + 1) * self.batch_size]
                all_points, all_normals, = [], []
                all_labels, all_primitives, all_parameters = [], [], []

                for idx in range(self.batch_size):
                    data_file = os.path.join(self.prefix, "{:05d}.h5".format(current_batch_idx[idx]))
                    with h5py.File(data_file, 'r') as hf:
                        points = np.array(hf.get("points"))
                        normals = np.array(hf.get("normals"))
                        labels = np.array(hf.get("labels"))
                        primitives = np.array(hf.get("prim"))
                        primitive_param = np.array(hf.get("T_param"))

                    all_points.append(points)
                    all_normals.append(normals)
                    all_labels.append(labels)
                    all_primitives.append(primitives)
                    all_parameters.append(primitive_param)

                all_points = np.stack(all_points, 0)
                all_normals = np.stack(all_normals, 0)
                all_labels = np.stack(all_labels, 0)
                all_primitives = np.stack(all_primitives, 0)
                all_parameters = np.stack(all_parameters, 0)

                if if_normal_noise:
                    noise = all_normals * np.clip(np.random.randn(1, all_points.shape[1], 1) * 0.01, a_min=-0.01,
                                                  a_max=0.01)
                    all_points = all_points + noise.astype(np.float32)

                # VOTE
                B, N, _ = all_points.shape
                all_cluster_vote = np.zeros((B, N, 3))
                for i in range(B):
                    one_unique_cluster = np.unique(all_labels[i])
                    one_point = all_points[i]
                    one_label = all_labels[i]
                    for c in one_unique_cluster:
                        cluster_idx = np.where(one_label == c)
                        cluster_point = np.array(one_point[cluster_idx])
                        center = 0.5 * (cluster_point.min(0) + cluster_point.max(0))
                        all_cluster_vote[i, cluster_idx, :] = center - cluster_point

                return_items = [all_points, all_labels, all_normals, all_primitives, all_parameters, all_cluster_vote, current_batch_idx]

                yield return_items

    def normalize_points(self, points, normals, anisotropic=False):
        points = points - np.mean(points, 0, keepdims=True)
        noise = normals * np.clip(np.random.randn(points.shape[0], 1) * 0.01, a_min=-0.01, a_max=0.01)
        points = points + noise.astype(np.float32)

        S, U = self.pca_numpy(points)
        smallest_ev = U[:, np.argmin(S)]
        R = self.rotation_matrix_a_to_b(smallest_ev, np.array([1, 0, 0]))
        # rotate input points such that the minor principal
        # axis aligns with x axis.
        points = (R @ points.T).T
        normals = (R @ normals.T).T
        std = np.max(points, 0) - np.min(points, 0)
        if anisotropic:
            points = points / (std.reshape((1, 3)) + EPS)
        else:
            points = points / (np.max(std) + EPS)
        return points.astype(np.float32), normals.astype(np.float32)

    def rotation_matrix_a_to_b(self, A, B):
        """
        Finds rotation matrix from vector A in 3d to vector B
        in 3d.
        B = R @ A
        """
        cos = np.dot(A, B)
        sin = np.linalg.norm(np.cross(B, A))
        u = A
        v = B - np.dot(A, B) * A
        v = v / (np.linalg.norm(v) + EPS)
        w = np.cross(B, A)
        w = w / (np.linalg.norm(w) + EPS)
        F = np.stack([u, v, w], 1)
        G = np.array([[cos, -sin, 0],
                      [sin, cos, 0],
                      [0, 0, 1]])
        # B = R @ A
        try:
            R = F @ G @ np.linalg.inv(F)
        except:
            R = np.eye(3, dtype=np.float32)
        return R

    def pca_numpy(self, X):
        S, U = np.linalg.eig(X.T @ X)
        return S, U


if __name__ == '__main__':
    from read_config import Config
    from torch.utils.data import DataLoader
    import open3d as o3d
    from src.dataset import generator_iter
    from src.our_vis import *

    config = Config("/home/zhuhan/Code/ProjectMarch/last_chance/parsenet-codebase/configs/config_parsenet_normals.yml")

    dataset = Dataset(
        config.batch_size,
        config.num_train,
        config.num_val,
        config.num_test,
        prefix="/home/zhuhan/Code/ProjectMarch/dataset/hpnet/",
        primitives=True,
        normals=True,
    )

    get_val_data = dataset.get_val(align_canonical=True, anisotropic=False, if_normal_noise=True)
    val_loader = generator_iter(get_val_data, int(1e10))
    get_val_data = iter(DataLoader(val_loader, batch_size=1, shuffle=False, collate_fn=lambda x: x, num_workers=2, pin_memory=False,))

    get_train_data = dataset.get_train(randomize=True, augment=True, align_canonical=True, anisotropic=False, if_normal_noise=True)
    train_loader = generator_iter(get_train_data, int(1e10))
    get_train_data = iter(DataLoader(train_loader, batch_size=1, shuffle=False, collate_fn=lambda x: x, num_workers=2, pin_memory=False,))

    for i in range(10):
        points, labels, normals, primitives, parameters, index = next(get_train_data)[0]

        print(i)

        # pcd=o3d.geometry.PointCloud()
        # pcd.points=o3d.utility.Vector3dVector(points[0])
        # pcd.normals=o3d.utility.Vector3dVector(normals[0])
        # pcd.colors=o3d.utility.Vector3dVector(colorful_by_cluster(labels[0]))
        # o3d.visualization.draw_geometries([pcd])
