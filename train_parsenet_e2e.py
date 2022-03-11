import gc
import logging
import numpy as np
import os
import sys
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'dataset'))
sys.path.append(os.path.join(BASE_DIR, 'tools'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,5,6,7,8"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import torch.optim as optim
import torch.utils.data
import traceback
from tensorboard_logger import configure, log_value
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dataset_separate import ABCDataset

from configs.read_config import Config
from models.ParseNet import PrimitivesEmbeddingDGCNGn
from utils.residual_utils import Evaluation
from models.segment_loss import (
    EmbeddingLoss,
    primitive_loss,
)

np.set_printoptions(precision=3)
config = Config("/home/zhuhan/Code/relationCNN/configs/config_parsenet_normals.yml")
model_name = config.model_name.format(config.batch_size, )

print(model_name)
time_stemp = time.strftime("%m-%d=%H:%M", time.localtime())

# don't need to record log when DEBUG
is_debug = False
if not is_debug:
    # configure("logs/tensorboard_e2e/{}_ep{}_{}".format(config.comment, config.epochs, time_stemp), flush_secs=5)
    configure(config.log_dir, flush_secs=5)

    # log will print to "console" and "log file"
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(asctime)s:%(name)s:%(message)s")
    file_handler = logging.FileHandler(
        config.log_dir + "/{}_{}.log".format(model_name, time_stemp), mode="w"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(handler)

Loss = EmbeddingLoss(margin=1.0, if_mean_shift=False)

model = PrimitivesEmbeddingDGCNGn(embedding=True, emb_size=128, primitives=True, num_primitives=10,
                                  loss_function=Loss.triplet_loss, mode=config.mode, num_channels=6, )
model = torch.nn.DataParallel(model, )
model.to("cuda")
optimizer = optim.Adam(model.parameters(), lr=config.lr)

if config.preload_model:
    pretrain_model = os.path.join(config.pretrain_model_path, config.pretrain_model_name)
    pretrain_data = torch.load(pretrain_model)
    start_ep = pretrain_data['epoch']
    model.load_state_dict(pretrain_data['model_dict'])
    try:
        optimizer.load_state_dict(pretrain_data['optimizer_dict'])
    except Exception as e:
        print(e)

    logger.info("let 's use {} as pretrain model and start from EP {}".format(config.pretrain_model_name, start_ep))

evaluation = Evaluation()
scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10, verbose=True, min_lr=1e-4)

if torch.cuda.device_count() > 1:
    logger.info("multi-gpu")
    alt_gpu = 1
else:
    logger.info("one gpu")
    alt_gpu = 0

DATA_PATH = config.dataset_path
TRAIN_DATASET = config.train_data
TEST_DATASET = config.test_data


# Init datasets and dataloaders
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


train_dataset = ABCDataset(DATA_PATH, TRAIN_DATASET, config=config, skip=config.train_skip)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, \
                                               shuffle=False, worker_init_fn=my_worker_init_fn)

test_dataset = ABCDataset(DATA_PATH, TEST_DATASET, config=config, skip=config.val_skip)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=config.batch_size, \
                                              shuffle=False, worker_init_fn=my_worker_init_fn)

# prev_test_loss = 1e4
prev_test_id_iou = 0
prev_test_type_iou = 0
logger.info("started training!")
lamb = 0.1
# no updates to the bn
model.eval()
batch_len = config.num_train // config.train_skip // config.batch_size

for e in range(config.epochs):
    train_id_ious = []
    train_type_ious = []
    train_losses = []
    train_emb_losses = []
    train_type_losses = []
    train_res_losses = []
    train_relation_losses = []
    n_loss = None

    # flag=True
    for batch_idx, batch_data_label in enumerate(train_dataloader):
        if batch_idx <= 50:
            continue
        # if '03640' not in batch_data_label['index'] and flag:
        #     continue
        # flag=False
        logger.info("batch_idx: {} index: {}".format(batch_idx, batch_data_label['index']))

        optimizer.zero_grad()
        losses = 0
        torch.cuda.empty_cache()
        t1 = time.time()

        for key in batch_data_label:
            if not isinstance(batch_data_label[key], list):
                batch_data_label[key] = batch_data_label[key].cuda()

        gc.collect()

        l = np.arange(10000)
        np.random.shuffle(l)
        l = l[0:config.num_points]

        points = (batch_data_label['gt_pc']).float().cuda()
        normals = (batch_data_label['gt_normal']).float().cuda()
        labels = batch_data_label['I_gt']
        primitives_ = batch_data_label['T_gt']

        points = points[:, l]
        labels = labels[:, l]
        normals = normals[:, l]
        primitives = primitives_[:, l]

        input = torch.cat([points, normals], 2)
        embedding, primitives_log_prob, embed_loss = model(input.permute(0, 2, 1), labels, True)
        torch.cuda.empty_cache()

        device = torch.device("cuda:{}".format(alt_gpu))
        relation_dict = {"orth": batch_data_label["gt_orthogonal"].to(device), "parall": batch_data_label["gt_parallel"].to(device),
                         "align": batch_data_label["gt_alignment"].to(device), "len": batch_data_label["gt_len"].to(device)}
        res_loss, relation_loss, _ = evaluation.fitting_loss(
            embedding.permute(0, 2, 1).to(torch.device("cuda:{}".format(alt_gpu))),  # (bs, 8k, 128)
            points.to(torch.device("cuda:{}".format(alt_gpu))),  # (bs, 8k, 3)
            normals.to(torch.device("cuda:{}".format(alt_gpu))),  # (bs, 8k, 3)
            labels,  # (bs, 8k,)
            primitives,  # (bs, 8k,)
            primitives_log_prob.to(torch.device("cuda:{}".format(alt_gpu))),  # (bs, 10, 8k,)
            relation_dict,
            quantile=0.025, iterations=config.ms_iter, lamb=lamb, batch_idx=batch_idx
        )

        res_loss[0] = res_loss[0].to(torch.device("cuda:0"))
        relation_loss = relation_loss.to(torch.device("cuda:0"))
        id_iou, type_iou = res_loss[3:]
        embed_loss = torch.mean(embed_loss)
        type_loss = primitive_loss(primitives_log_prob, primitives)
        relation_loss = relation_loss * config.relation_loss_weight
        loss = embed_loss + type_loss + res_loss[0] + relation_loss  # loss = embed_loss + p_loss + 1 * res_loss[0]

        torch.cuda.empty_cache()
        loss.backward()

        optimizer.step()
        torch.cuda.empty_cache()

        train_id_ious.append(id_iou)
        train_type_ious.append(type_iou)
        train_losses.append(loss.data.cpu().numpy())
        train_emb_losses.append(embed_loss.data.cpu().numpy())
        train_type_losses.append(type_loss.data.cpu().numpy())
        train_res_losses.append(res_loss[0].data.cpu().numpy())
        train_relation_losses.append(relation_loss.data.cpu().numpy())


        log_value("id_iou", id_iou, batch_idx + e * batch_len)
        log_value("type_iou", type_iou, batch_idx + e * batch_len)
        log_value("all_loss", loss, batch_idx + e * batch_len)
        log_value("embed_loss", embed_loss, batch_idx + e * batch_len)
        log_value("type_loss", type_loss, batch_idx + e * batch_len)
        log_value("res_loss", res_loss[0], batch_idx + e * batch_len)
        log_value("relation_loss", relation_loss, batch_idx + e * batch_len)

        if batch_idx % config.log_interval == 0 and batch_idx != 0:
            # print("batch_idx: ",batch_idx)
            logger.info(
                "Epoch: {:03d} iter: {:04d} | id-iou: {:06f} type-iou: {:06f} | embed loss: {:06f}, type loss: {:06f}, res loss:{:06f} relation_loss:{:06f}".format(
                    e, batch_idx, id_iou, type_iou, embed_loss, type_loss, res_loss[0].data.cpu().numpy(), relation_loss[0].detach().cpu().numpy()))

        del loss, embed_loss, type_loss, res_loss

    save_dict = {
        'epoch': e + 1,
        'optimizer_dict': optimizer.state_dict()
    }
    try:  # with nn.DataParallel() the net is added as a submodule of DataParallel
        save_dict['model_state_dict'] = model.module.state_dict()
    except:
        save_dict['model_state_dict'] = model.state_dict()
    torch.save(save_dict, os.path.join(config.log_dir, "ckpt_ep{}.tar".format(e)))

    # === eval one ep ===
    test_type_ious = []
    test_id_ious = []
    test_losses = []
    test_emb_losses = []
    test_type_losses = []
    test_res_losses = []
    test_relation_losses = []

    model.eval()
    torch.cuda.empty_cache()

    for batch_idx, batch_data_label in enumerate(test_dataloader):
        t1 = time.time()

        for key in batch_data_label:
            if not isinstance(batch_data_label[key], list):
                batch_data_label[key] = batch_data_label[key].cuda()

        l = np.arange(10000)
        np.random.shuffle(l)
        l = l[0:config.num_points]
        points = (batch_data_label['gt_pc']).float().cuda()
        normals = (batch_data_label['gt_normal']).float().cuda()
        labels = batch_data_label['I_gt']
        primitives_ = batch_data_label['T_gt']

        points = points[:, l]
        labels = labels[:, l]
        normals = normals[:, l]
        primitives = primitives_[:, l]

        with torch.no_grad():
            input = torch.cat([points, normals], 2)
            embedding, primitives_log_prob, embed_loss = model(input.permute(0, 2, 1), labels, True)

            try:
                res_loss, relation_loss, _ = evaluation.fitting_loss(
                    embedding.permute(0, 2, 1).to(torch.device("cuda:{}".format(alt_gpu))),
                    points.to(torch.device("cuda:{}".format(alt_gpu))),
                    normals.to(torch.device("cuda:{}".format(alt_gpu))),
                    labels,
                    primitives,
                    primitives_log_prob.to(torch.device("cuda:{}".format(alt_gpu))),
                    quantile=0.025, iterations=10, lamb=1.0, eval=True,
                )
            except Exception:
                traceback.print_exc()
                loss = embed_loss
                loss.backward()
                print("some exception in while testing")
                continue

        id_iou, type_iou = res_loss[3:]
        embed_loss = torch.mean(embed_loss)
        type_loss = primitive_loss(primitives_log_prob, primitives)
        res_loss = res_loss[0].to(torch.device("cuda:0"))
        relation_loss = relation_loss * config.relation_loss_weight
        loss = embed_loss + type_loss + res_loss + relation_loss

        test_type_ious.append(type_iou)
        test_id_ious.append(id_iou)
        test_emb_losses.append(embed_loss.data.cpu().numpy())
        test_type_losses.append(type_loss.data.cpu().numpy())
        test_res_losses.append(res_loss.item())
        test_relation_losses.append(relation_loss.data.cpu().numpy())
        test_losses.append(loss.data.cpu().numpy())
        torch.cuda.empty_cache()

    print("\n")
    logger.info(
        "Epoch: {}/{} => TrID-IoU:{}, TsID-IoU:{}, TrTYPE-IoU:{}, TsTYPE-IoU:{} |  TrLoss:{}, TsLoss:{}, TrTypeLoss:{}, TsTypeLoss:{}, TrEemLoss:{}, TsEemLoss:{}, ".format(
            e,
            config.epochs,
            np.mean(train_id_ious),
            np.mean(test_id_ious),
            np.mean(train_type_ious),
            np.mean(test_type_ious),

            np.mean(train_losses),
            np.mean(test_losses),
            np.mean(train_type_losses),
            np.mean(test_type_losses),
            np.mean(train_emb_losses),
            np.mean(test_emb_losses),

        )
    )
    log_value("train id iou", np.mean(train_id_ious), e)
    log_value("test id iou", np.mean(test_id_ious), e)

    log_value("train emb loss", np.mean(train_emb_losses), e)
    log_value("test emb loss", np.mean(test_emb_losses), e)

    log_value("train res loss", np.mean(train_res_losses), e)
    log_value("test res loss", np.mean(train_res_losses), e)

    log_value("train type iou", np.mean(train_type_ious), e)
    log_value("test type iou", np.mean(test_type_ious), e)

    log_value("train relation loss", np.mean(train_relation_losses), e)
    log_value("test relation loss", np.mean(test_relation_losses), e)

    scheduler.step(np.mean(test_res_losses))

    if np.mean(test_id_ious) > prev_test_id_iou:
        prev_test_id_iou = np.mean(test_id_ious)
        logger.info("id iou improvement, saving models at epoch: {}".format(e))
        save_dict = {
            'epoch': e + 1,
            'optimizer_dict': optimizer.state_dict()
        }
        try:  # with nn.DataParallel() the net is added as a submodule of DataParallel
            save_dict['model_state_dict'] = model.module.state_dict()
        except:
            save_dict['model_state_dict'] = model.state_dict()
        torch.save(save_dict, os.path.join(config.log_dir, "ckpt_ep{}_id{:8f}.tar".format(e, np.mean(test_id_ious))))

    if np.mean(test_type_ious) > prev_test_type_iou:
        prev_test_type_iou = np.mean(test_type_ious)
        logger.info("type iou improvement, saving models at epoch: {}".format(e))
        save_dict = {
            'epoch': e + 1,
            'optimizer_dict': optimizer.state_dict()
        }
        try:  # with nn.DataParallel() the net is added as a submodule of DataParallel
            save_dict['model_state_dict'] = model.module.state_dict()
        except:
            save_dict['model_state_dict'] = model.state_dict()
        torch.save(save_dict, os.path.join(config.log_dir, "ckpt_ep{}_type{:8f}.tar".format(e, np.mean(test_type_ious))))


