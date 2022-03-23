"""
This scrip trains model to predict per point primitive type.
"""
import json
import logging
import os
import time
import sys
from shutil import copyfile
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import numpy as np
import torch.optim as optim
import torch.utils.data
from tensorboard_logger import configure, log_value
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from read_config import Config
# from src.PointNet import PrimitivesEmbeddingDGCNGn
from models.relationcnn import PrimitivesEmbeddingDGCNGn
from src.dataset import generator_iter
from src.new_dataset_segments import Dataset
from src.segment_loss import (
    EmbeddingLoss,
    evaluate_miou,
    primitive_loss
)
from models.rel_loss import (
    compute_normal_loss,
    compute_vote_loss,
    compute_param_loss
)

config = Config("/home/zhuhan/Code/ProjectMarch/last_chance/parsenet-codebase/configs/config_parsenet_normals.yml")
model_name = config.model_path.format(
    config.comment,
    config.batch_size,
    config.lr,
    config.num_train,
    config.num_test,
    config.loss_weight,
    config.mode,
)
print(model_name)
configure("logs/tensorboard/{}".format(model_name), flush_secs=5)

userspace = os.path.dirname(os.path.abspath(__file__))

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("%(asctime)s:%(name)s:%(message)s")
file_handler = logging.FileHandler(
    "/home/zhuhan/Code/ProjectMarch/last_chance/parsenet-codebase/logs/not_e2e/{}.log".format(model_name), mode="w"
)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(handler)


if_normals = config.normals
if_normal_noise = True

Loss = EmbeddingLoss(margin=1.0, if_mean_shift=False)
if config.mode == 0:
    # Just using points for training
    model = PrimitivesEmbeddingDGCNGn(embedding=True, emb_size=128, primitives=True,
                                      num_primitives=10, loss_function=Loss.triplet_loss, mode=config.mode, num_channels=3,)
elif config.mode == 5:
    # Using points and normals for training
    model = PrimitivesEmbeddingDGCNGn(emb_size=128, primitives=True, num_primitives=10,
                                      loss_function=Loss.triplet_loss,mode=5, num_channels=6,)

if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
model.cuda()


dataset = Dataset(config.batch_size, config.num_train, config.num_val, config.num_test, prefix=config.dataset_path, primitives=True, normals=True,)
get_train_data = dataset.get_train(randomize=True, augment=True, align_canonical=True, anisotropic=False, if_normal_noise=if_normal_noise)
get_val_data = dataset.get_val(align_canonical=True, anisotropic=False, if_normal_noise=if_normal_noise)
loader = generator_iter(get_train_data, int(1e10))
get_train_data = iter(DataLoader(loader, batch_size=1, shuffle=False, collate_fn=lambda x: x, num_workers=2, pin_memory=False,))
loader = generator_iter(get_val_data, int(1e10))
get_val_data = iter(DataLoader(loader, batch_size=1, shuffle=False, collate_fn=lambda x: x, num_workers=2, pin_memory=False,))

optimizer = optim.Adam(model.parameters(), lr=config.lr)
os.makedirs("logs/trained_models/{}/".format(model_name), exist_ok=True)
scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4, verbose=True, min_lr=1e-4)

prev_test_loss = 1e4
print("started training!")
for e in range(config.epochs):
    train_emb_losses = []
    train_prim_losses = []
    train_normal_losses = []
    train_param_losses = []
    train_vote_losses = []
    train_iou = []
    train_losses = []
    model.train()

    # this is used for gradient accumulation because of small gpu memory.
    num_iter = 3
    for train_b_id in range(config.num_train // config.batch_size):
        # if train_b_id == 2:
        #     break
        optimizer.zero_grad()
        losses = 0
        ious = 0
        p_losses = 0
        embed_losses = 0
        normal_losses = 0
        param_losses = 0
        vote_losses = 0
        torch.cuda.empty_cache()
        for _ in range(num_iter):
            points, labels, normals, primitives, parameters, vote, item_index = next(get_train_data)[0]
            l = np.arange(10000)
            np.random.shuffle(l)
            # randomly sub-sampling points to increase robustness to density and
            # saving gpu memory
            rand_num_points = 8000
            l = l[0:rand_num_points]
            points = points[:, l]
            labels = labels[:, l]
            normals = normals[:, l]
            primitives = primitives[:, l]
            parameters = parameters[:, l]
            vote = vote[:, l]
            points = torch.from_numpy(points).float().cuda()
            normals = torch.from_numpy(normals).float().cuda()
            primitives = torch.from_numpy(primitives.astype(np.int64)).cuda()
            parameters = torch.from_numpy(parameters).cuda()
            vote = torch.from_numpy(vote).cuda()

            if if_normals:
                # ms_feat, primitives_log_prob, embed_loss, normal_per_point, param_per_point, offset_per_point
                input = torch.cat([points, normals], 2)
                embedding, primitives_log_prob, embed_loss, normal_per_point, param_per_point, offset_per_point = model(
                    input.permute(0, 2, 1), torch.from_numpy(labels).cuda(), True
                )
            else:
                embedding, primitives_log_prob, embed_loss, normal_per_point, param_per_point, offset_per_point = model(
                    points.permute(0, 2, 1), torch.from_numpy(labels).cuda(), True
                )
            embed_loss = torch.mean(embed_loss)
            p_loss = primitive_loss(primitives_log_prob, primitives)
            normal_loss = compute_normal_loss(normal_per_point, normals)
            normal_loss = config.normal_weight * normal_loss
            param_loss = compute_param_loss(param_per_point, primitives, parameters)
            param_loss = config.param_weight * param_loss
            vote_loss = compute_vote_loss(offset_per_point, vote,)
            vote_loss = config.vote_weight * vote_loss

            iou = evaluate_miou(
                primitives.data.cpu().numpy(),
                primitives_log_prob.permute(0, 2, 1).data.cpu().numpy(),
            )
            loss = embed_loss + p_loss + normal_loss + param_loss + vote_loss
            loss.backward()

            losses += loss.data.cpu().numpy() / num_iter
            p_losses += p_loss.data.cpu().numpy() / num_iter
            ious += iou / num_iter
            embed_losses += embed_loss.data.cpu().numpy() / num_iter
            normal_losses += normal_loss.data.cpu().numpy() / num_iter
            param_losses += param_loss.data.cpu().numpy() / num_iter
            vote_losses += vote_loss.data.cpu().numpy() / num_iter

        optimizer.step()
        train_iou.append(ious)
        train_losses.append(losses)
        train_prim_losses.append(p_losses)
        train_emb_losses.append(embed_losses)
        train_normal_losses.append(normal_losses)
        train_param_losses.append(param_losses)
        train_vote_losses.append(vote_losses)
        time_stemp = time.strftime("%m-%d=%H:%M", time.localtime())
        print(
            "\r{} Epoch: {} iter: {}, prim loss: {}, emb loss: {}, normal loss: {}, param loss: {}, vote loss: {}, iou: {}".format(
                time_stemp, e, train_b_id, p_loss, embed_losses, normal_losses, param_losses, vote_losses, iou
            ),
            end="",
        )
        if train_b_id % 10 == 0:
            logger.info(
                "\r{} Epoch: {} iter: {}, prim loss: {}, emb loss: {}, normal loss: {}, param loss: {}, vote loss: {}, iou: {}".format(
                time_stemp, e, train_b_id, p_loss, embed_losses, normal_losses, param_losses, vote_losses, iou
                ),
            )

        log_value("iou", iou, train_b_id + e * (config.num_train // config.batch_size))
        log_value("embed_loss", embed_losses, train_b_id + e * (config.num_train // config.batch_size),)
        log_value("normal_losses", normal_losses, train_b_id + e * (config.num_train // config.batch_size),)
        log_value("param_losses", param_losses, train_b_id + e * (config.num_train // config.batch_size),)
        log_value("vote_losses", vote_losses, train_b_id + e * (config.num_train // config.batch_size),)

    test_emb_losses = []
    test_prim_losses = []
    test_vote_losses = []
    test_param_losses = []
    test_normal_losses = []
    test_losses = []
    test_iou = []
    model.eval()

    for val_b_id in range(config.num_val // config.batch_size - 1):
        # if val_b_id==2:
        #     break
        points, labels, normals, primitives, parameters, vote, item_index = next(get_val_data)[0]
        l = np.arange(10000)
        np.random.shuffle(l)
        l = l[0:8000]

        points = points[:, l]
        labels = labels[:, l]
        normals = normals[:, l]
        primitives = primitives[:, l]
        parameters = parameters[:, l]
        vote = vote[:, l]
        points = torch.from_numpy(points).float().cuda()
        normals = torch.from_numpy(normals).float().cuda()
        primitives = torch.from_numpy(primitives.astype(np.int64)).cuda()
        parameters = torch.from_numpy(parameters).cuda()
        vote = torch.from_numpy(vote).cuda()

        with torch.no_grad():
            if if_normals:
                input = torch.cat([points, normals], 2)
                embedding, primitives_log_prob, embed_loss, normal_per_point, param_per_point, offset_per_point = model(
                    input.permute(0, 2, 1), torch.from_numpy(labels).cuda(), True
                )
            else:
                embedding, primitives_log_prob, embed_loss, normal_per_point, param_per_point, offset_per_point = model(
                    points.permute(0, 2, 1), torch.from_numpy(labels).cuda(), True
                )

        embed_loss = torch.mean(embed_loss)
        p_loss = primitive_loss(primitives_log_prob, primitives)
        normal_loss = compute_normal_loss(normal_per_point, normals)
        normal_loss = config.normal_weight * normal_loss
        param_loss = compute_param_loss(param_per_point, primitives, parameters)
        param_loss = config.param_weight * param_loss
        vote_loss = compute_vote_loss(offset_per_point, vote, )
        vote_loss = config.vote_weight * vote_loss

        loss = embed_loss + p_loss + normal_loss + param_loss + vote_loss
        iou = evaluate_miou(
            primitives.data.cpu().numpy(),
            primitives_log_prob.permute(0, 2, 1).data.cpu().numpy(),
        )
        test_iou.append(iou)
        test_prim_losses.append(p_loss.data.cpu().numpy())
        test_emb_losses.append(embed_loss.data.cpu().numpy())
        test_normal_losses.append(normal_loss.data.cpu().numpy())
        test_param_losses.append(param_loss.data.cpu().numpy())
        test_vote_losses.append(vote_loss.data.cpu().numpy())
        test_losses.append(loss.data.cpu().numpy())

    torch.cuda.empty_cache()
    print("\n")
    logger.info(
        "Epoch: {}/{} => TrL:{}, TsL:{}, TrP:{}, TsP:{}, TrE:{}, TsE:{}, TrNor:{}, TsNor:{},TrVot:{}, TsVot:{},TrPara:{}, TsPara:{}, TrI:{}, TsI:{}".format(
            e,
            config.epochs,
            np.mean(train_losses),
            np.mean(test_losses),
            np.mean(train_prim_losses),
            np.mean(test_prim_losses),
            np.mean(train_emb_losses),
            np.mean(test_emb_losses),

            np.mean(train_normal_losses),
            np.mean(test_normal_losses),
            np.mean(train_vote_losses),
            np.mean(test_vote_losses),
            np.mean(train_param_losses),
            np.mean(test_param_losses),

            np.mean(train_iou),
            np.mean(test_iou),
        )
    )
    log_value("train iou", np.mean(train_iou), e)
    log_value("test iou", np.mean(test_iou), e)
    log_value("train emb loss", np.mean(train_emb_losses), e)
    log_value("test emb loss", np.mean(test_emb_losses), e)

    log_value("train normal loss", np.mean(train_normal_losses), e)
    log_value("test normal loss", np.mean(test_normal_losses), e)
    log_value("train vote loss", np.mean(train_vote_losses), e)
    log_value("test vote loss", np.mean(test_vote_losses), e)
    log_value("train parameter loss", np.mean(train_param_losses), e)
    log_value("test parameter loss", np.mean(test_param_losses), e)

    torch.save(
        model.state_dict(),
        "logs/trained_models/{}/ep{}.pth".format(model_name, e),
    )
    torch.save(
        optimizer.state_dict(),
        "logs/trained_models/{}/ep{}_optimizer.pth".format(model_name, e),
    )
    scheduler.step(np.mean(test_emb_losses))
    if prev_test_loss > np.mean(test_emb_losses):
        logger.info("improvement, saving model at epoch: {}".format(e))
        prev_test_loss = np.mean(test_emb_losses)
        torch.save(
            model.state_dict(),
            "logs/trained_models/{}/best_{}.pth".format(model_name, e),
        )
        torch.save(
            optimizer.state_dict(),
            "logs/trained_models/{}/best_{}_optimizer.pth".format(model_name, e),
        )
