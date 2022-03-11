import os
import h5py
import numpy as np
from collections import Counter
from augment_utils import rotate_perturbation_point_cloud, jitter_point_cloud, shift_point_cloud, \
    random_scale_point_cloud, rotate_point_cloud
import open3d as o3d
from models.segment_loss import (
    EmbeddingLoss,
)
from utils.fitting_utils import (
    weights_normalize,
    match,
)
import torch
from torch.utils.data import Dataset
from utils.residual_utils import Evaluation, fit_one_shape_torch
from utils.segment_utils import to_one_hot
from scipy import stats
from configs.read_config import Config
from torch.utils import data

EPS = np.finfo(np.float32).eps


class ABCDataset(data.Dataset):
    def __init__(self, root, filename, config, skip=1, fold=1):

        self.root = root
        self.data_path = open(os.path.join(root, filename), 'r')
        self.opt = config
        self.augment_routines = [
            rotate_perturbation_point_cloud, jitter_point_cloud,
            shift_point_cloud, random_scale_point_cloud, rotate_point_cloud
        ]

        if 'train' in filename:
            self.augment = self.opt.augment
            self.if_normal_noise = self.opt.if_normal_noise
        else:
            self.augment = 0
            self.if_normal_noise = 0

        self.data_list = [item.strip() for item in self.data_path.readlines()]
        self.skip = skip

        self.data_list = self.data_list[::self.skip]
        self.tru_len = len(self.data_list)
        self.len = self.tru_len * fold

    def __getitem__(self, index):

        ret_dict = {}
        index = index % self.tru_len

        data_file = os.path.join(self.root, self.data_list[index] + '.h5')

        with h5py.File(data_file, 'r') as hf:
            points = np.array(hf.get("points"))
            labels = np.array(hf.get("labels"))
            normals = np.array(hf.get("normals"))
            primitives = np.array(hf.get("prim"))
            primitive_param = np.array(hf.get("T_param"))

        relation_file = os.path.join(self.root, 'relations', self.data_list[index] + '_relations.h5')
        with h5py.File(relation_file, 'r') as hf:
            gt_len = np.array(hf.get("gt_len"))
            gt_parallel = np.array(hf.get("gt_parallel"))
            gt_orthogonal = np.array(hf.get("gt_orthogonal"))
            gt_alignment = np.array(hf.get("gt_alignment"))

        ret_dict["gt_len"] = np.array(gt_len)
        ret_dict["gt_parallel"] = np.array([[-1, -1]] * 100)
        if len(gt_parallel) != 0: ret_dict["gt_parallel"][:gt_len[0]] = gt_parallel[:gt_len[0]]
        ret_dict["gt_orthogonal"] = np.array([[-1, -1]] * 100)
        if len(gt_orthogonal) != 0: ret_dict["gt_orthogonal"][:gt_len[1]] = gt_orthogonal[:gt_len[1]]
        ret_dict["gt_alignment"] = np.array([[-1, -1]] * 100)
        if len(gt_alignment) != 0: ret_dict["gt_alignment"][:gt_len[2]] = gt_alignment[:gt_len[2]]

        if self.augment:
            points = self.augment_routines[np.random.choice(np.arange(5))](points[None, :, :])[0]

        if self.if_normal_noise:
            noise = normals * np.clip(np.random.randn(points.shape[0], 1) * 0.01, a_min=-0.01, a_max=0.01)
            points = points + noise.astype(np.float32)

        ret_dict['gt_pc'] = points
        ret_dict['gt_normal'] = normals
        ret_dict['T_gt'] = primitives.astype(int)
        ret_dict['T_param'] = primitive_param
        ret_dict['index'] = self.data_list[index]

        old = labels
        count = list(Counter(labels).keys())
        mapper = np.ones([max(count) + 1]) * -1
        mapper[count] = np.arange(len(count))
        new = mapper[old]
        ret_dict['I_gt'] = new.astype(int)

        # get each points offset
        num_points = points.shape[0]
        unique_cluster = np.unique(labels)
        cluster_vote = np.zeros((num_points, 3))
        for c in unique_cluster:
            cluster_idx = np.where(labels == c)
            cluster_point = np.array(points[cluster_idx])
            center = 0.5 * (cluster_point.min(0) + cluster_point.max(0))
            cluster_vote[cluster_idx, :] = center - cluster_point
        ret_dict['V_gt'] = cluster_vote

        return ret_dict

    def __len__(self):
        return self.len


if __name__ == '__main__':
    config_path = os.path.join("/home/zhuhan/Code/relationCNN/configs/config_parsenet_normals.yml")
    config = Config(config_path)

    DATA_PATH = config.dataset_path
    TRAIN_DATASET = config.train_data
    TEST_DATASET = config.test_data


    # Init datasets and dataloaders
    def my_worker_init_fn(worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id)


    Dataset = ABCDataset

    train_dataset = Dataset(DATA_PATH, TRAIN_DATASET, config=config, skip=config.train_skip)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, \
                                                   shuffle=False,
                                                   worker_init_fn=my_worker_init_fn)

    for batch_idx, batch_label in enumerate(train_dataloader):
        if batch_idx % 10 == 0:
            print(batch_idx)
        for i in range(config.batch_size):
            count = sorted(list(Counter(batch_label["I_gt"][i].cpu().numpy()).keys()))
            count_len = len(count)
            count_min = min(count)
            count_max = max(count)
            assert count_max == count_min + count_len - 1
                # print("cluster id", count)
                # old = batch_label["I_gt"][i].cpu().numpy()
                # mapper = np.ones([max(count) + 1]) * -1
                # mapper[count] = np.arange(len(count))
                # new = mapper[old]
            # if sorted(count)[0] != 0:
            #     cnt = 0
            #     for idx in range(10000):
            #         if batch_label["I_gt"][i][idx] not in count:
            #             cnt += 1
            #     print("cnt", cnt)

