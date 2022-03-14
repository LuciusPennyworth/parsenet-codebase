from open3d import *
import h5py
import sys
import logging
import json
import os
from src.utils import chamfer_distance_single_shape
from src.segment_utils import sample_from_collection_of_mesh
from shutil import copyfile
import numpy as np
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from src.PointNet import PrimitivesEmbeddingDGCNGn
from matplotlib import pyplot as plt
from src.utils import visualize_uv_maps, visualize_fitted_surface
from src.utils import chamfer_distance
from read_config import Config
from src.utils import fit_surface_sample_points
from src.dataset_segments import Dataset
from torch.utils.data import DataLoader
from src.utils import chamfer_distance
from src.segment_loss import EmbeddingLoss
from src.segment_utils import cluster
import time
from src.segment_loss import (
    EmbeddingLoss,
    primitive_loss,
    evaluate_miou,
)
from src.segment_utils import to_one_hot, SIOU_matched_segments
from src.utils import visualize_point_cloud_from_labels, visualize_point_cloud
from src.dataset import generator_iter
from src.mean_shift import MeanShift
from src.segment_utils import SIOU_matched_segments
from src.residual_utils import Evaluation


def continuous_labels(labels_):
    new_labels = np.zeros_like(labels_)
    for index, value in enumerate(np.sort(np.unique(labels_))):
        new_labels[labels_ == value] = index
    return new_labels


# Use only one gpu.
os.environ["CUDA_VISIBLE_DEVICES"] = "9"
config = Config("/home/zhuhan/Code/ProjectMarch/last_chance/parsenet-codebase/configs/config_test_parsenet_normals.yml")
if_normals = config.normals

userspace = ""
Loss = EmbeddingLoss(margin=1.0)

if config.mode == 0:
    # Just using points for training
    model = PrimitivesEmbeddingDGCNGn(
        embedding=True,
        emb_size=128,
        primitives=True,
        num_primitives=10,
        loss_function=Loss.triplet_loss,
        mode=config.mode,
        num_channels=3,
    )
elif config.mode == 5:
    # Using points and normals for training
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

dataset = Dataset(
    config.batch_size,
    config.num_train,
    config.num_val,
    config.num_test,
    normals=True,
    primitives=True,
    if_train_data=False,
    prefix=config.dataset_path,
)

get_test_data = dataset.get_test(align_canonical=True, anisotropic=False, if_normal_noise=True)

loader = generator_iter(get_test_data, int(1e10))
get_test_data = iter(DataLoader(loader, batch_size=1, shuffle=False, collate_fn=lambda x: x, num_workers=0, pin_memory=False,))

# os.makedirs(userspace + "logs/results/{}/results/".format(config.pretrain_model_path), exist_ok=True)

evaluation = Evaluation()
alt_gpu = 0
model.eval()

iterations = 50
quantile = 0.015

model.load_state_dict(
    torch.load("/home/zhuhan/Code/ProjectMarch/last_chance/parsenet-codebase/logs/pretrained_models/parsenet_with_normals.pth")
)
test_res = []
test_s_iou = []
test_p_iou = []
PredictedLabels = []
PredictedPrims = []

for val_b_id in range(config.num_test // config.batch_size - 1):
    points_, labels, normals, primitives_ = next(get_test_data)[0]
    points = Variable(torch.from_numpy(points_.astype(np.float32))).cuda()
    normals = torch.from_numpy(normals).cuda()

    # with torch.autograd.detect_anomaly():
    with torch.no_grad():
        if if_normals:
            input = torch.cat([points, normals], 2)
            embedding, primitives_log_prob, embed_loss = model(
                input.permute(0, 2, 1), torch.from_numpy(labels).cuda(), True
            )
        else:
            embedding, primitives_log_prob, embed_loss = model(
                points.permute(0, 2, 1), torch.from_numpy(labels).cuda(), True
            )
    pred_primitives = torch.max(primitives_log_prob[0], 0)[1].data.cpu().numpy()
    embedding = torch.nn.functional.normalize(embedding[0].T, p=2, dim=1)
    _, _, cluster_ids = evaluation.guard_mean_shift(
        embedding, quantile, iterations, kernel_type="gaussian"
    )
    weights = to_one_hot(cluster_ids, np.unique(cluster_ids.data.data.cpu().numpy()).shape[0])
    cluster_ids = cluster_ids.data.cpu().numpy()

    s_iou, p_iou, _, _ = SIOU_matched_segments(labels[0], cluster_ids, pred_primitives, primitives_[0], weights,)
    time_stemp = time.strftime("%m-%d=%H:%M", time.localtime())
    print(time_stemp, " ", val_b_id, s_iou, p_iou)
    test_s_iou.append(s_iou)
    test_p_iou.append(p_iou)
    PredictedLabels.append(cluster_ids)
    PredictedPrims.append(pred_primitives)


print("===id_iou:{:06f} type_iou:{:06f}===".format(np.mean(test_s_iou), np.mean(test_p_iou)))

# with h5py.File(userspace + "logs/results/{}/results/".format(config.pretrain_model_path) + "predictions.h5", "w") as hf:
#     hf.create_dataset(name="seg_id", data=np.stack(PredictedLabels, 0))
#     hf.create_dataset(name="pred_primitives", data=np.stack(PredictedPrims, 0))

test_cluster_ids = np.stack(PredictedLabels, 0)
test_pred_primitives = np.stack(PredictedPrims, 0).astype(np.int32)

evaluation = Evaluation()
root_path = "/home/zhuhan/Code/ProjectMarch/dataset/parsenet/data/shapes/test_data.h5"
with h5py.File(root_path, "r") as hf:
    test_points = np.array(hf.get("points"))
    test_labels = np.array(hf.get("labels"))
    test_normals = np.array(hf.get("normals"))
    test_primitives = np.array(hf.get("prim"))


test_s_iou = []
test_p_iou = []
s_k_1s = []
s_k_2s = []
p_k_1s = []
p_k_2s = []
s_ks = []
p_ks = []
test_cds = []
bw = 0.01

for i in range(len(test_cluster_ids)):
    points = test_points[i].astype(np.float32)
    normals = test_normals[i].astype(np.float32)

    labels = test_labels[i].astype(np.int32)
    labels = continuous_labels(labels)

    cluster_ids = test_cluster_ids[i].astype(np.int32)
    cluster_ids = continuous_labels(cluster_ids)
    weights = to_one_hot(cluster_ids, np.unique(cluster_ids).shape[0])

    points, normals = dataset.normalize_points(points, normals)
    torch.cuda.empty_cache()
    with torch.no_grad():
        # if_visualize=True, will give you all segments
        # if_sample=True will return segments as trimmed meshes
        # if_optimize=True will optimize the spline surface patches
        _, _, newer_pred_mesh = evaluation.residual_eval_mode(
            torch.from_numpy(points).cuda(),
            torch.from_numpy(normals).cuda(),
            labels,
            cluster_ids,
            test_primitives[i],
            test_pred_primitives[i],
            weights.T,
            bw,
            sample_points=True,
            if_optimize=False,
            if_visualize=True,
            epsilon=0.1)

    torch.cuda.empty_cache()
    s_iou, p_iou, _, _ = SIOU_matched_segments(labels, cluster_ids, test_pred_primitives[i], test_primitives[i], weights,)

    test_s_iou.append(s_iou)
    test_p_iou.append(p_iou)

    try:
        Points = sample_from_collection_of_mesh(newer_pred_mesh)
    except Exception as e:
        print("error in sample_from_collection_of_mesh method", e)
        continue
    cd1 = chamfer_distance_single_shape(torch.from_numpy(Points).cuda(), torch.from_numpy(points).cuda(), sqrt=True,
                                        one_side=True, reduce=False)
    cd2 = chamfer_distance_single_shape(torch.from_numpy(points).cuda(), torch.from_numpy(Points).cuda(), sqrt=True,
                                        one_side=True, reduce=False)

    s_k_1s.append(torch.mean((cd1 < 0.01).float()).item())
    s_k_2s.append(torch.mean((cd1 < 0.02).float()).item())
    s_ks.append(torch.mean(cd1).item())
    p_k_1s.append(torch.mean((cd2 < 0.01).float()).item())
    p_k_2s.append(torch.mean((cd2 < 0.02).float()).item())
    p_ks.append(torch.mean(cd2).item())
    test_cds.append((s_ks[-1] + p_ks[-1]) / 2.0)

    results = {"sk_1": s_k_1s[-1],
               "sk_2": s_k_2s[-1],
               "sk": s_ks[-1],
               "pk_1": p_k_1s[-1],
               "pk_2": p_k_2s[-1],
               "pk": p_ks[-1],
               "cd": test_cds[-1],
               "p_iou": p_iou,
               "s_iou": s_iou}

    print(i, s_iou, p_iou, test_cds[-1])

print("Test CD: {}, Test p cover: {}, Test s cover: {}".format(np.mean(test_cds), np.mean(s_ks), np.mean(p_ks)))
print("iou seg: {}, iou prim type: {}".format(np.mean(test_s_iou), np.mean(test_p_iou)))











