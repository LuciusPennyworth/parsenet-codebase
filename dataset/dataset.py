"""
This script defines dataset loading for the segmentation task on ABC dataset.
"""
# os.environ["CUDA_VISIBLE_DEVICES"] = "8"
import os, sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
import h5py
import numpy as np
from collections import Counter
# import open3d as o3d
import torch
from scipy import stats

from models.segment_loss import (
    EmbeddingLoss,
)
from augment_utils import rotate_perturbation_point_cloud, jitter_point_cloud, shift_point_cloud, \
    random_scale_point_cloud, rotate_point_cloud
from utils.fitting_utils import (
    weights_normalize,
    match,
)
from utils.residual_utils import Evaluation, fit_one_shape_torch
from utils.segment_utils import to_one_hot

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
                 prefix="",
                 position_encode=False,
                 encode_level=10):
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
        self.position_encode = position_encode
        self.encode_level = encode_level

        if if_train_data:
            print("prefix: " + prefix)
            with h5py.File(prefix + "data/shapes/train_data.h5", "r") as hf:
                train_points = np.array(hf.get("points"))
                train_labels = np.array(hf.get("labels"))
                if normals:
                    train_normals = np.array(hf.get("normals"))
                if primitives:
                    train_primitives = np.array(hf.get("prim"))
            train_points = train_points[0:train_size].astype(np.float32)
            train_labels = train_labels[0:train_size]
            self.train_normals = train_normals[0:train_size].astype(np.float32)
            self.train_primitives = train_primitives[0:train_size]
            means = np.mean(train_points, 1)
            means = np.expand_dims(means, 1)

            self.train_points = (train_points - means)
            self.train_labels = train_labels

        with h5py.File(prefix + "data/shapes/val_data.h5", "r") as hf:
            val_points = np.array(hf.get("points"))
            val_labels = np.array(hf.get("labels"))
            if normals:
                val_normals = np.array(hf.get("normals"))
            if primitives:
                val_primitives = np.array(hf.get("prim"))

        with h5py.File(prefix + "data/shapes/test_data.h5", "r") as hf:
        # with h5py.File(prefix + "data/shapes/mini_test.h5", "r") as hf:
            test_points = np.array(hf.get("points"))
            test_labels = np.array(hf.get("labels"))
            if normals:
                test_normals = np.array(hf.get("normals"))
            if primitives:
                test_primitives = np.array(hf.get("prim"))

        val_points = val_points[0:val_size].astype(np.float32)
        val_labels = val_labels[0:val_size]

        test_points = test_points[0:test_size].astype(np.float32)
        test_labels = test_labels[0:test_size]

        if normals:
            self.val_normals = val_normals[0:val_size].astype(np.float32)
            self.test_normals = test_normals[0:test_size].astype(np.float32)

        if primitives:
            self.val_primitives = val_primitives[0:val_size]
            self.test_primitives = test_primitives[0:test_size]

        means = np.mean(test_points, 1)
        means = np.expand_dims(means, 1)
        self.test_points = (test_points - means)
        self.test_labels = test_labels

        means = np.mean(val_points, 1)
        means = np.expand_dims(means, 1)
        self.val_points = (val_points - means)
        self.val_labels = val_labels

    def get_train(self, randomize=False, augment=False, anisotropic=False, align_canonical=False,
                  if_normal_noise=False):
        train_size = self.train_points.shape[0]

        l = np.arange(train_size)
        if randomize:
            np.random.shuffle(l)
        train_points = self.train_points[l]
        train_labels = self.train_labels[l]

        if self.normals:
            train_normals = self.train_normals[l]
        if self.primitives:
            train_primitives = self.train_primitives[l]

        while True:
            for i in range(train_size // self.batch_size):
                points = train_points[i * self.batch_size:(i + 1) *
                                                          self.batch_size]
                if self.normals:
                    normals = train_normals[i * self.batch_size:(i + 1) * self.batch_size]

                if augment:
                    points = self.augment_routines[np.random.choice(np.arange(5))](points)

                if if_normal_noise:
                    normals = train_normals[i * self.batch_size:(i + 1) * self.batch_size]

                    noise = normals * np.clip(np.random.randn(1, points.shape[1], 1) * 0.01, a_min=-0.01, a_max=0.01)
                    points = points + noise.astype(np.float32)

                labels = train_labels[i * self.batch_size:(i + 1) * self.batch_size]

                for j in range(self.batch_size):
                    if align_canonical:
                        S, U = self.pca_numpy(points[j])
                        smallest_ev = U[:, np.argmin(S)]
                        R = self.rotation_matrix_a_to_b(smallest_ev, np.array([1, 0, 0]))
                        # rotate input points such that the minor principal
                        # axis aligns with x axis.
                        points[j] = (R @ points[j].T).T

                        if self.normals:
                            normals[j] = (R @ normals[j].T).T

                        std = np.max(points[j], 0) - np.min(points[j], 0)
                        if anisotropic:
                            points[j] = points[j] / (std.reshape((1, 3)) + EPS)
                        else:
                            points[j] = points[j] / (np.max(std) + EPS)

                if self.position_encode:
                    points = self.positional_encoding(points, self.encode_level)

                return_items = [points, labels]
                if self.normals:
                    return_items.append(normals)
                else:
                    return_items.append(None)

                if self.primitives:
                    primitives = train_primitives[i * self.batch_size:(i + 1) * self.batch_size]
                    return_items.append(primitives)
                else:
                    return_items.append(None)

                # compute votes *AFTER* augmentation
                num_points = points.shape[1]
                point_votes = np.zeros([self.batch_size, num_points, 3])
                for j in range(self.batch_size):
                    shape_points = points[j][:, :3]
                    shape_cluster = labels[j]
                    shape_prim = primitives[j]

                    unique_cluster = np.unique(shape_cluster)
                    cluster_vote = np.zeros((len(shape_points), 3))
                    for cluster in unique_cluster:
                        one_cluster_idx = np.where(shape_cluster == cluster)
                        cluster_prim = self.get_cluster_prim(Counter(shape_prim[one_cluster_idx]))
                        assert cluster_prim != -1 and cluster_prim != 11
                        # if cluster_prim not in [1, 3, 4, 5]:
                        #     continue
                        one_cluster_point = np.array(shape_points[one_cluster_idx])
                        center = 0.5 * (one_cluster_point.min(0) + one_cluster_point.max(0))
                        cluster_vote[one_cluster_idx, :] = center - one_cluster_point
                    point_votes[j] = cluster_vote
                    # print("{} % points in current shape have vote GT.".format(100 * float(sum(point_votes[j, :, 0]!=0)/len(shape_points)), "0.2f"))
                    # check_vote_gt(shape_points, cluster_vote)

                return_items.append(point_votes)
                yield return_items

    def get_test(self, randomize=False, anisotropic=False, align_canonical=False, if_normal_noise=False):
        test_size = self.test_points.shape[0]
        batch_size = self.batch_size

        while True:
            for i in range(test_size // batch_size):
                points = self.test_points[i * self.batch_size:(i + 1) * self.batch_size]
                labels = self.test_labels[i * self.batch_size:(i + 1) * self.batch_size]
                if self.normals:
                    normals = self.test_normals[i * self.batch_size:(i + 1) * self.batch_size]
                if if_normal_noise and self.normals:
                    normals = self.test_normals[i * self.batch_size:(i + 1) * self.batch_size]
                    noise = normals * np.clip(np.random.randn(1, points.shape[1], 1) * 0.01, a_min=-0.01, a_max=0.01)
                    points = points + noise.astype(np.float32)

                for j in range(self.batch_size):
                    if align_canonical:
                        S, U = self.pca_numpy(points[j])
                        smallest_ev = U[:, np.argmin(S)]
                        R = self.rotation_matrix_a_to_b(smallest_ev, np.array([1, 0, 0]))
                        # rotate input points such that the minor principal
                        # axis aligns with x axis.
                        points[j] = (R @ points[j].T).T
                        if self.normals:
                            normals[j] = (R @ normals[j].T).T

                        std = np.max(points[j], 0) - np.min(points[j], 0)
                        if anisotropic:
                            points[j] = points[j] / (std.reshape((1, 3)) + EPS)
                        else:
                            points[j] = points[j] / (np.max(std) + EPS)

                if self.position_encode:
                    points = self.positional_encoding(points, self.encode_level)

                return_items = [points, labels]
                if self.normals:
                    return_items.append(normals)
                else:
                    return_items.append(None)

                if self.primitives:
                    primitives = self.test_primitives[i * self.batch_size:(i + 1) * self.batch_size]
                    return_items.append(primitives)
                else:
                    return_items.append(None)

                # compute votes *AFTER* augmentation
                num_points = points.shape[1]
                point_votes = np.zeros([self.batch_size, num_points, 3])
                for j in range(self.batch_size):
                    shape_points = points[j][:, :3]
                    shape_cluster = labels[j]
                    shape_prim = primitives[j]

                    unique_cluster = np.unique(shape_cluster)
                    cluster_vote = np.zeros((len(shape_points), 3))
                    for cluster in unique_cluster:
                        one_cluster_idx = np.where(shape_cluster == cluster)
                        cluster_prim = self.get_cluster_prim(Counter(shape_prim[one_cluster_idx]))
                        assert cluster_prim != -1 and cluster_prim != 11
                        # if cluster_prim not in [1, 3, 4, 5]:
                        #     continue
                        one_cluster_point = np.array(shape_points[one_cluster_idx])
                        center = 0.5 * (one_cluster_point.min(0) + one_cluster_point.max(0))
                        cluster_vote[one_cluster_idx, :] = center - one_cluster_point
                    point_votes[j] = cluster_vote
                return_items.append(point_votes)

                yield return_items

    def get_val(self, randomize=False, anisotropic=False, align_canonical=False, if_normal_noise=False):
        val_size = self.val_points.shape[0]
        batch_size = self.batch_size

        while True:
            for i in range(val_size // batch_size):
                points = self.val_points[i * self.batch_size:(i + 1) *
                                                             self.batch_size]
                labels = self.val_labels[i * self.batch_size:(i + 1) * self.batch_size]
                if self.normals:
                    normals = self.val_normals[i * self.batch_size:(i + 1) *
                                                                   self.batch_size]
                if if_normal_noise and self.normals:
                    normals = self.val_normals[i * self.batch_size:(i + 1) *
                                                                   self.batch_size]
                    noise = normals * np.clip(np.random.randn(1, points.shape[1], 1) * 0.01, a_min=-0.01, a_max=0.01)
                    points = points + noise.astype(np.float32)

                for j in range(self.batch_size):
                    if align_canonical:
                        S, U = self.pca_numpy(points[j])
                        smallest_ev = U[:, np.argmin(S)]
                        R = self.rotation_matrix_a_to_b(smallest_ev, np.array([1, 0, 0]))
                        # rotate input points such that the minor principal
                        # axis aligns with x axis.
                        points[j] = (R @ points[j].T).T
                        if self.normals:
                            normals[j] = (R @ normals[j].T).T

                        std = np.max(points[j], 0) - np.min(points[j], 0)
                        if anisotropic:
                            points[j] = points[j] / (std.reshape((1, 3)) + EPS)
                        else:
                            points[j] = points[j] / (np.max(std) + EPS)

                if self.position_encode:
                    points = self.positional_encoding(points, self.encode_level)

                return_items = [points, labels]
                if self.normals:
                    return_items.append(normals)
                else:
                    return_items.append(None)

                if self.primitives:
                    primitives = self.val_primitives[i * self.batch_size:(i + 1) * self.batch_size]
                    return_items.append(primitives)
                else:
                    return_items.append(None)

                # compute votes *AFTER* augmentation
                num_points = points.shape[1]
                point_votes = np.zeros([self.batch_size, num_points, 3])
                for j in range(self.batch_size):
                    shape_points = points[j][:, :3]
                    shape_cluster = labels[j]
                    shape_prim = primitives[j]

                    unique_cluster = np.unique(shape_cluster)
                    cluster_vote = np.zeros((len(shape_points), 3))
                    for cluster in unique_cluster:
                        one_cluster_idx = np.where(shape_cluster == cluster)
                        cluster_prim = self.get_cluster_prim(Counter(shape_prim[one_cluster_idx]))
                        assert cluster_prim != -1 and cluster_prim != 11
                        # if cluster_prim not in [1, 3, 4, 5]:
                        #     continue
                        one_cluster_point = np.array(shape_points[one_cluster_idx])
                        center = 0.5 * (one_cluster_point.min(0) + one_cluster_point.max(0))
                        cluster_vote[one_cluster_idx, :] = center - one_cluster_point
                    point_votes[j] = cluster_vote
                return_items.append(point_votes)

                yield return_items

    def positional_encoding(self, points, N_freqs):
        periodic_fns = [torch.sin, torch.cos]
        embed_fns = [lambda x: x]
        max_freq = N_freqs - 1

        freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)
        for freq in freq_bands:
            for p_fn in periodic_fns:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))

        return torch.cat([fn(torch.Tensor(points)) for fn in embed_fns], -1)

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

    def get_cluster_prim(self, dict):
        cluster_prim = -1
        most_common_points = -1
        for k, v in dict.items():
            if v > most_common_points:
                cluster_prim = k
                most_common_points = v
        return cluster_prim


def get_train_dataset(config):
    from torch.utils.data import DataLoader
    from dataset_utils import generator_iter

    dataset = Dataset(
        config.batch_size,
        config.num_train,
        config.num_val,
        config.num_test,
        primitives=True,
        normals=True,
        prefix="/home/zhuhan/ParseNet/Dataset/parsenet-data/"
    )
    get_train_data = dataset.get_train(
        randomize=False, augment=True, align_canonical=True, anisotropic=False,
        if_normal_noise=True
    )
    loader = generator_iter(get_train_data, int(1e10))
    get_train_data = iter(
        DataLoader(
            loader,
            batch_size=1,
            shuffle=False,
            collate_fn=lambda x: x,
            num_workers=2,
            pin_memory=False,
        )
    )

    return get_train_data


def get_test_dataset(config):
    from torch.utils.data import DataLoader
    from dataset_utils import generator_iter

    dataset = Dataset(
        config.batch_size,
        config.num_train,
        config.num_val,
        config.num_test,
        primitives=True,
        normals=True,
        prefix="/home/zhuhan/ParseNet/Dataset/parsenet-data/",
        encode_level=config.encode_level
    )
    get_test_data = dataset.get_test(
        randomize=False, align_canonical=True, anisotropic=False, if_normal_noise=False)
    loader = generator_iter(get_test_data, int(1e10))
    get_test_data = iter(
        DataLoader(
            loader,
            shuffle=False,
            collate_fn=lambda x: x,
            num_workers=1,
            pin_memory=False,
        )
    )
    return get_test_data


def check_vote_gt(config):
    get_train_data = get_train_dataset()
    for train_b_id in range(config.num_train // config.batch_size):
        points, labels, normals, primitives, votes = next(get_train_data)[0]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        vote_point = np.zeros((2 * points.shape[0], 3))
        vote_line = np.zeros((points.shape[0], 2))
        for i in range(points.shape[0]):
            vote_point[2 * i] = points[i]
            vote_point[2 * i + 1] = points[i] + votes[i]
            vote_line[i] = [2 * i, 2 * i + 1]

        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(vote_point),
            lines=o3d.utility.Vector2iVector(vote_line)
        )

        o3d.visualization.draw_geometries([pcd, ])
        o3d.visualization.draw_geometries([pcd, line_set])
        o3d.visualization.draw_geometries([line_set])

    return


def gen_param_gt_by_train(config):
    Loss = EmbeddingLoss(margin=1.0, if_mean_shift=False)
    userspace = "/home/zhuhan/ParseNet/Dataset/parsenet-data/"
    bw = 0.01
    evaluation = Evaluation()
    iterations = 50
    quantile = 0.015
    if torch.cuda.device_count() > 1:
        print("multi-gpu")
        alt_gpu = 1
    else:
        print("one gpu")
        alt_gpu = 0

    model = PrimitivesEmbeddingDGCNGn(
        embedding=True,
        emb_size=128,
        primitives=True,
        num_primitives=10,
        loss_function=Loss.triplet_loss,
        mode=config.mode,
        num_channels=6,
    )
    model = torch.nn.DataParallel(model, device_ids=[0])
    model.cuda()
    model.load_state_dict(
        torch.load(userspace + "logs/pretrained_models/" + config.pretrain_model_path)
    )

    get_train_data = get_train_dataset()
    for train_b_id in range(config.num_train // config.batch_size):
        points, labels, normals, primitives, votes = next(get_train_data)[0]

        with torch.no_grad():
            input = torch.cat([points, normals], 2)
            embedding, primitives_log_prob, embed_loss = model(
                input.permute(0, 2, 1), torch.from_numpy(labels).cuda(), True
            )

            for batch_idx in range(config.batch_size):
                pred_primitives = torch.max(primitives_log_prob[batch_idx], 0)[1].data.cpu().numpy()
                embedding_one = torch.nn.functional.normalize(embedding[batch_idx].T, p=2, dim=1)
                _, _, cluster_ids = evaluation.guard_mean_shift(
                    embedding_one, quantile, iterations, kernel_type="gaussian"
                )
            weights = to_one_hot(cluster_ids, np.unique(cluster_ids.data.data.cpu().numpy()).shape[
                0])

            # if_visualize=True, will give you all segments
            # if_sample=True will return segments as trimmed meshes
            # if_optimize=True will optimize the spline surface patches
            _, parameters, newer_pred_mesh = evaluation.residual_eval_mode(
                torch.from_numpy(points).cuda(),
                torch.from_numpy(normals).cuda(),
                test_labels,  # gt cluster
                pred_cluster_ids,  # gen cluster ids
                test_primitives[i],  # gt prim
                pred_primitives[i],  # gen prim
                weights.T,
                bw,
                sample_points=True,
                if_optimize=False,
                if_visualize=True,
                epsilon=0.1)


def take_some_pics():
    from configs.read_config import Config
    import os
    from tools.vis_util import colorful_by_prim, tr_to_np

    config = os.path.join(os.getcwd(), "../configs/config_parsenet_normals.yml")
    config = Config(config)
    test_data = get_test_dataset(config)

    for test_id in range(config.num_test // config.batch_size):
        print("current val_id : {}".format(test_id))
        points_, labels, normals, primitives_, votes = next(test_data)[0]

        for idx in range(config.batch_size):
            point = tr_to_np(points_[idx][:, :3])
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(point)
            cols = colorful_by_prim(primitives_[idx])
            pcd.colors = o3d.utility.Vector3dVector(cols)
            o3d.visualization.draw_geometries([pcd])
            print()


def gen_param_gt():
    train_data = "/home/zhuhan/ParseNet/Dataset/parsenet-data/data/shapes/train_data.h5"
    prefix = "/home/zhuhan/ParseNet/Dataset/parsenet-data/"
    dataset = Dataset(
        1,
        24000,
        4000,
        4000,
        normals=True,
        primitives=True,
        if_train_data=True,
        prefix=prefix
    )
    evaluation = Evaluation()
    with h5py.File(train_data, "r") as hf:
        # N x 3
        train_points = np.array(hf.get("points"))
        # N x 1
        train_labels = np.array(hf.get("labels"))
        # N x 3
        train_normals = np.array(hf.get("normals"))
        # N x 1
        train_primitives = np.array(hf.get("prim"))

    train_param = []
    for i in range(90):
        print("shape {}".format(i))
        bw = 0.01
        points = train_points[i].astype(np.float32)
        normals = train_normals[i].astype(np.float32)
        primitive = train_primitives[i].astype(np.float32)
        label = train_labels[i].astype(np.int32)
        label = continuous_labels(label)
        weights = to_one_hot(label, np.unique(label).shape[0])

        with torch.no_grad():
            points = torch.from_numpy(points).cuda()
            normals = torch.from_numpy(normals).cuda()
            label = torch.from_numpy(label).cuda()

            if not isinstance(label, np.ndarray):
                label = label.data.cpu().numpy()

            # weights = weights.data.cpu().numpy()
            weights = (
                to_one_hot(label,
                           np.unique(label).shape[0], device_id=weights.get_device()).data.cpu().numpy().T)

            rows, cols, unique_target, unique_pred = match(label, label)
            data = []
            all_segments = []
            for index, i in enumerate(unique_pred):
                # TODO some labels might be missing from unique_pred
                gt_indices_i = label == cols[index]
                pred_indices_i = label == i
                if (np.sum(gt_indices_i) == 0) or (np.sum(pred_indices_i) == 0):
                    continue

                l = stats.mode(primitive[pred_indices_i])[0]  # get the most common type in the same cluster
                data.append(
                    [
                        points[pred_indices_i],
                        normals[pred_indices_i],
                        l,
                        points[gt_indices_i],
                        pred_indices_i,
                        (index, i),
                    ]
                )
            all_segments.append([data, weights, bw])

            shape_params = np.zeros((len(points), 22))
            for index, data in enumerate(all_segments):
                if index >= 1:
                    print("index >= 1")
                data_, weights_first, bw = data

                weights_first = torch.from_numpy(weights_first.astype(np.float32)).cuda()
                weights = weights_normalize(weights_first, float(bw))
                weights = torch.transpose(weights, 1, 0)
                weights = to_one_hot(
                    torch.max(weights, 1)[1].data.cpu().numpy(), weights.shape[1],
                )

                gt_points, recon_points = fit_one_shape_torch(
                    data_,
                    evaluation.fitter,
                    weights,
                    bw,
                    eval=True,
                    sample_points=False,
                    if_optimize=True,
                    if_visualize=False,
                )

                # save gt dataset in here
                fitting_params = evaluation.fitter.fitting.parameters
                for idx, d in enumerate(data_):
                    # points, normals, labels, gpoints, segment_indices, part_index = d
                    _, _, labels, _, segment_indices, _ = d

                    if labels not in [1, 3, 4, 5]:
                        continue
                    if fitting_params[idx] == None:
                        continue
                    # sphere
                    if labels == 5:
                        shape_params[segment_indices, :3] = fitting_params[idx][1].squeeze(0).cpu().numpy()
                        shape_params[segment_indices, 3] = fitting_params[idx][2].cpu().numpy()
                    # plane
                    elif labels == 1:
                        shape_params[segment_indices, 4:7] = fitting_params[idx][1].squeeze(1).cpu().numpy()
                        shape_params[segment_indices, 7] = fitting_params[idx][2].cpu().numpy()
                    # cone
                    elif labels == 3:
                        shape_params[segment_indices, 8:11] = fitting_params[idx][1].squeeze(0).cpu().numpy()
                        shape_params[segment_indices, 11:14] = fitting_params[idx][2].squeeze(1).cpu().numpy()
                        shape_params[segment_indices, 14] = fitting_params[idx][3].cpu().numpy()
                    # cylinder
                    elif labels == 4:
                        shape_params[segment_indices, 15:18] = fitting_params[idx][1].squeeze(1).cpu().numpy()
                        shape_params[segment_indices, 18:21] = fitting_params[idx][2].squeeze(0).cpu().numpy()
                        shape_params[segment_indices, 21] = fitting_params[idx][3].cpu().numpy()
                    else:
                        assert 1 == 0

            train_param.append(shape_params)

    return np.stack(train_param, 0)


def continuous_labels(labels_):
    new_labels = np.zeros_like(labels_)
    for index, value in enumerate(np.sort(np.unique(labels_))):
        new_labels[labels_ == value] = index
    return new_labels


if __name__ == '__main__':
    # gt_param_idx = []
    # train_param = gen_param_gt()
    # with h5py.File("/home/zhuhan/ParseNet/Dataset/parsenet-data/data/shapes/params.h5", "w") as hf:
    #     hf.create_dataset(name="params", data=train_param)
    # print("finish.")
    take_some_pics()
