import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
import torch
import numpy as np
from torch.utils import data
from scipy import stats
import h5py
import random
from collections import Counter
from augment_utils import rotate_perturbation_point_cloud, jitter_point_cloud, \
    shift_point_cloud, random_scale_point_cloud, rotate_point_cloud
from utils.residual_utils import point_2_line_distance_torch

EPS = np.finfo(np.float32).eps


class ABCDataset(data.Dataset):
    def __init__(self, root, filename, opt, skip=1, fold=1):

        self.root = root
        self.data_path = open(os.path.join(root, filename), 'r')
        self.opt = opt
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

        # relation_file = os.path.join(self.root, 'relations', self.data_list[index] + '_relations.h5')
        # with h5py.File(relation_file, 'r') as hf:
        #     ds_gt_len = np.array(hf.get("gt_len"))
        #     ds_gt_parallel = np.array(hf.get("gt_parallel"))
        #     ds_gt_orthogonal = np.array(hf.get("gt_orthogonal"))
        #     ds_gt_alignment = np.array(hf.get("gt_alignment"))

        if self.augment:
            points = self.augment_routines[np.random.choice(np.arange(5))](points[None, :, :])[0]

        if self.if_normal_noise:
            noise = normals * np.clip(
                np.random.randn(points.shape[0], 1) * 0.01,
                a_min=-0.01,
                a_max=0.01)
            points = points + noise.astype(np.float32)

        ret_dict['gt_pc'] = points
        ret_dict['gt_normal'] = normals
        ret_dict['T_gt'] = primitives.astype(int)
        ret_dict['T_param'] = primitive_param

        # set small number primitive as background
        counter = Counter(labels)
        mapper = np.ones([labels.max() + 1]) * -1
        keys = [k for k, v in counter.items() if v > 100]
        if len(keys):
            mapper[keys] = np.arange(len(keys))
        label = mapper[labels]
        ret_dict['I_gt'] = label.astype(int)
        clean_primitives = np.ones_like(primitives) * -1
        valid_mask = label != -1
        clean_primitives[valid_mask] = primitives[valid_mask]
        ret_dict['T_gt'] = clean_primitives.astype(int)

        ret_dict['index'] = self.data_list[index]

        small_idx = label == -1
        full_labels = label
        full_labels[small_idx] = labels[small_idx] + len(keys)
        ret_dict['I_gt_clean'] = full_labels.astype(int)

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
        ret_dict['valid_mask'] = valid_mask

        gt_len = np.array([-1, -1, -1])  # parallel \ orthogonal \ alignment
        # 1. collect each cluster's parameters
        example = ret_dict
        all_param = {"spline_id": []}
        Points = {}
        sphere_list, cone_cylinder_list = [], []
        for c_id in np.unique(example["I_gt"]):
            if c_id == -1:  # if c_id==-1, means this cluster is background
                continue
            c_idx = example["I_gt"] == c_id
            type = stats.mode(example["T_gt"][c_idx])[0]
            if type not in [1, 3, 4, 5]:
                continue
            points = torch.Tensor(example["gt_pc"][c_idx])
            param = torch.Tensor(example["T_param"][c_idx][0])

            if type == 1:  # plane
                c_para = param[4:8]
                all_param[c_id] = ['plane', c_para[:3], c_para[3]]
                Points[c_id] = points
            elif type == 3:  # cone
                c_para = param[15:22]
                all_param[c_id] = ['cone', c_para[:3], c_para[3:6], c_para[6]]
                Points[c_id] = points
                cone_cylinder_list.append(c_id)
            elif type == 4:  # cylinder
                c_para = param[8:15]
                all_param[c_id] = ['cylinder', c_para[:3], c_para[3:6], c_para[6]]
                Points[c_id] = points
                cone_cylinder_list.append(c_id)
            elif type == 5:  # sphere
                c_para = param[0:4]
                all_param[c_id] = ['sphere', c_para[:3], c_para[3]]
                Points[c_id] = points
                sphere_list.append(c_id)
            else:
                all_param["spline_id"].extend(c_id)
        # distance = res_loss.residual_loss(Points, all_param)

        # 2. find parallel and orthogonal
        # 2.1 collect all primitive pairs which might be parallel or orthogonal
        candidates_direction = {}
        for k, v in all_param.items():
            if k != 'spline_id' and v[0] in ['plane', 'cone', 'cylinder']:
                if v[0] == 'plane':
                    direction_vec = v[1]
                if v[0] == 'cone':
                    direction_vec = v[1]
                if v[0] == 'cylinder':
                    direction_vec = v[1]
                candidates_direction[k] = direction_vec
        pll_keys = list(candidates_direction.keys())
        orientation_candidate_pair = [sorted([pll_keys[i], pll_keys[j]])
                                      for i in range(len(pll_keys)) for j in range(i + 1, len(pll_keys))]

        # 2.2 calculate and find whether they are parallel or orthogonal
        gt_parallel, gt_orthogonal, other_rel = [], [], []
        for candidate_pair in orientation_candidate_pair:
            i, j = candidate_pair
            i_vec, j_vec = candidates_direction[i], candidates_direction[j]
            dot = np.abs(i_vec.unsqueeze(0) @ j_vec.unsqueeze(1))
            if np.abs(dot - 1.0) < 0.0152:
                gt_parallel.append(candidate_pair)
                # display_particular_points(example["gt_pc"], example["I_gt"], candidate_pair, os.path.join(placement_path, "parallel", "{}.png".format(str(candidate_pair))))
            elif np.abs(dot) < 0.17:
                gt_orthogonal.append(candidate_pair)
                # display_particular_points(example["gt_pc"], example["I_gt"], candidate_pair, os.path.join(placement_path, "orthogonal", "{}.png".format(str(candidate_pair))))
            else:
                other_rel.append(candidate_pair)

        # 2.3 save result

        gt_len[0] = min(len(gt_parallel), 100)
        gt_len[1] = min(len(gt_orthogonal), 100)

        ret_dict["gt_parallel"] = np.array([[-1, -1]] * 100)
        if len(gt_parallel) != 0:
            ret_dict["gt_parallel"][:gt_len[0]] = gt_parallel[:gt_len[0]]

        ret_dict["gt_orthogonal"] = np.array([[-1, -1]] * 100)
        if len(gt_orthogonal) != 0:
            ret_dict["gt_orthogonal"][:gt_len[1]] = gt_orthogonal[:gt_len[1]]

        # 3. find axis alignment
        gt_alignment = []
        # 3.1 find axis alignment in parallel cone and cylinder
        for para_pair in gt_parallel:
            i, j = para_pair
            if all_param[i][0] in ['cone', 'cylinder'] and all_param[j][0] in ['cone', 'cylinder']:
                dist_i2j = point_2_line_distance_torch(all_param[i][2], all_param[j][2], all_param[j][1])
                dist_j2i = point_2_line_distance_torch(all_param[j][2], all_param[i][2], all_param[i][1])
                if min(dist_i2j, dist_j2i) < 0.01:
                    # display_particular_points(example["gt_pc"], example["I_gt"], para_pair, os.path.join(alignment_path, "{}.png".format(str(para_pair))))
                    gt_alignment.append(para_pair)
        # 3.2 find sphere center align to cone and cylinder
        for one_sphere in sphere_list:
            for one_cc in cone_cylinder_list:
                dist_s2c = point_2_line_distance_torch(all_param[one_sphere][1], all_param[one_cc][2],
                                                       all_param[one_cc][1])
                if dist_s2c < 0.01:
                    # display_particular_points(example["gt_pc"], example["I_gt"], [one_sphere, one_cc], os.path.join(alignment_path, "{}.png".format(str([one_sphere, one_cc]))))
                    gt_alignment.append(sorted([one_sphere, one_cc]))

        # 3.3 save result
        gt_len[2] = min(len(gt_alignment), 100)

        ret_dict["gt_alignment"] = np.array([[-1, -1]] * 100)
        if len(gt_alignment) != 0:
            ret_dict["gt_alignment"][:gt_len[2]] = gt_alignment[:gt_len[2]]

        ret_dict["gt_len"] = np.array(gt_len)

        # check(ret_dict, ds_gt_len, ds_gt_parallel, ds_gt_orthogonal, ds_gt_alignment)

        return ret_dict

    def __len__(self):
        return self.len


def check(ret_dict, ds_gt_len, ds_gt_parallel, ds_gt_orthogonal, ds_gt_alignment):
    gt_len = ret_dict['gt_len']
    l1, l2, l3 = gt_len
    gt_parallel = ret_dict['gt_parallel'][:l1]
    gt_orthogonal = ret_dict['gt_orthogonal'][:l2]
    gt_alignment = ret_dict['gt_alignment'][:l3]

    if l1 != len(ds_gt_parallel) or l2 != len(ds_gt_orthogonal) or l3 != len(ds_gt_alignment):
        print(gt_len)
        print(len(ds_gt_parallel), len(ds_gt_orthogonal), len(ds_gt_alignment))
        assert 1 == 2


def rename():
    dir_path = "/home/zhuhan/Code/ProjectMarch/dataset/hpnet/relations/"
    f = os.listdir(dir_path)
    for old in f:
        if old[-1] == '.':
            oldname = os.path.join(dir_path, old)
            newname = oldname[:-1]
            os.rename(oldname, newname)


if __name__ == '__main__':
    from option import build_option
    import time


    # Init datasets and dataloaders
    def my_worker_init_fn(worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id)


    opt = build_option()
    Dataset = ABCDataset

    # train_dataset = Dataset(opt.data_path, opt.train_dataset, opt=opt, skip=opt.train_skip, fold=opt.train_fold)
    test_dataset = Dataset(opt.data_path, opt.test_dataset, opt=opt, skip=opt.val_skip)

    # train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=False, worker_init_fn=my_worker_init_fn)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batch_size, \
                                                  shuffle=False, worker_init_fn=my_worker_init_fn)

    data_time = time.time()
    data_timer = 0
    save_dir = '/home/zhuhan/Code/ProjectMarch/dataset/hpnet/relations'
    for batch_idx, batch_data_label in enumerate(test_dataloader):
        # data_timer += (time.time() - data_time)
        if batch_idx % 10 == 0:
            print(batch_idx)

        for num, data_idx in enumerate(batch_data_label['index']):
            one_relation_len = batch_data_label['gt_len'][num]
            data_idx = int(data_idx)

            # with h5py.File(os.path.join(save_dir, '{:05d}_relations.h5').format(data_idx), 'w') as wf:
            #     wf.create_dataset("gt_len", data=one_relation_len.cpu().numpy())
            #     wf.create_dataset("gt_parallel",
            #                       data=batch_data_label['gt_parallel'][num][:one_relation_len[0]].cpu().numpy())
            #     wf.create_dataset("gt_orthogonal",
            #                       data=batch_data_label['gt_orthogonal'][num][:one_relation_len[1]].cpu().numpy())
            #     wf.create_dataset("gt_alignment",
            #                       data=batch_data_label['gt_alignment'][num][:one_relation_len[2]].cpu().numpy())

