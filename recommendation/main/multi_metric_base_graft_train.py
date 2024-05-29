# coding=utf-8
import collections
import os
import time
import json
import logging
import math
import argparse
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.multiprocessing as mp
from torch import nn

import model
from util.timer import Timer
from util import args_processing as ap
from util import consts
from util import env
from loader import multi_metric_meta_sequence_dataloader as sequence_dataloader
from util import new_metrics
import numpy as np
from thop import profile

from util import utils
utils.setup_seed(0)
# utils.setup_seed(1)

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables", type=str, help="Kernels configuration for CNN")
    parser.add_argument("--bucket", type=str, default=None, help="Bucket name for external storage")
    parser.add_argument("--dataset", type=str, default="alipay", help="Bucket name for external storage")

    parser.add_argument("--max_steps", type=int, help="Number of iterations before stopping")
    parser.add_argument("--snapshot", type=int, help="Number of iterations to dump model")
    parser.add_argument("--checkpoint_dir", type=str, help="Path of the checkpoint path")
    parser.add_argument("--learning_rate", type=str, default=0.001)
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    # parser.add_argument("--max_epoch", type=int, default=10, help="Max epoch")
    parser.add_argument("--max_epoch", type=int,  default=10, help="Max epoch")
    parser.add_argument("--num_loading_workers", type=int, default=4, help="Number of threads for loading")
    parser.add_argument("--model", type=str, help="model type")
    parser.add_argument("--init_checkpoint", type=str, default="", help="Path of the checkpoint path")
    parser.add_argument("--init_step", type=int, default=0, help="Path of the checkpoint path")

    parser.add_argument("--max_gradient_norm", type=float, default=0.)

    # If both of the two options are set, `model_config` is preferred
    parser.add_argument("--arch_config_path", type=str, default=None, help="Path of model configs")
    parser.add_argument("--arch_config", type=str, default=None, help="base64-encoded model configs")

    # graft
    parser.add_argument("--base_model_path", type=str, default=None, help="Path of the base model to be grafted")
    parser.add_argument("--graft_model_path", type=str, default=None, help="Path of the graft model to graft")
    parser.add_argument("--checkpoint_dir_1", type=str, default=None, help="Path of the checkpoint for model1")
    parser.add_argument("--checkpoint_dir_2", type=str, default=None, help="Path of the checkpoint for model2")
    parser.add_argument("--finetune", action="store_true", default=False, help="choose to finetune")
    parser.add_argument("--ft_epoch", type=int, default=0, help="number of epochs for finetuning")
    parser.add_argument("--finetune_path", type=str, default=None, help="path of the model to be finetuned")

    return parser.parse_known_args()[0]



# def predict(predict_dataset, model_obj, device, args, bucket, train_step, writer=None):

def predict(predict_dataset, model_obj, device, args, train_epoch, train_step, graft_param, finetune=False, writer=None):
    model_obj.eval()
    model_obj.to(device)

    timer = Timer()
    log_every = 200
    pred_list = []
    y_list = []
    buffer = []
    user_id_list = []
    for step, batch_data in enumerate(predict_dataset, 1):
        logits = model_obj({
            key: value.to(device)
            for key, value in batch_data.items()
            if key not in {consts.FIELD_USER_ID, consts.FIELD_LABEL, consts.FIELD_TRIGGER_SEQUENCE}
        }, graft_param, finetune=finetune)

        prob = torch.sigmoid(logits).detach().cpu().numpy()
        y = batch_data[consts.FIELD_LABEL].view(-1, 1)
        overall_auc, _, _, _ = new_metrics.calculate_overall_auc(prob, y)
        user_id_list.extend(np.array(batch_data[consts.FIELD_USER_ID].view(-1, 1)))
        pred_list.extend(prob)
        y_list.extend(np.array(y))

        buffer.extend(
            [int(user_id), float(score), float(label)]
            for user_id, score, label
            in zip(
                batch_data[consts.FIELD_USER_ID],
                prob,
                batch_data[consts.FIELD_LABEL]
            )
        )
        if step % log_every == 0:
            logger.info(
                "train_epoch={}, step={}, overall_auc={:5f}, speed={:2f} steps/s".format(
                    train_epoch, step, overall_auc, log_every / timer.tick(False)
                )
            )

    overall_auc, _, _, _ = new_metrics.calculate_overall_auc(np.array(pred_list), np.array(y_list))
    user_auc = new_metrics.calculate_user_auc(buffer)
    overall_logloss = new_metrics.calculate_overall_logloss(np.array(pred_list), np.array(y_list))
    user_ndcg5, user_hr5 = new_metrics.calculate_user_ndcg_hr(5, buffer)
    user_ndcg10, user_hr10 = new_metrics.calculate_user_ndcg_hr(10, buffer)
    user_ndcg20, user_hr20 = new_metrics.calculate_user_ndcg_hr(20, buffer)

    print("train_epoch={}, train_step={}, overall_auc={:5f}, user_auc={:5f}, overall_logloss={:5f}, "
          "user_ndcg5={:5f}, user_hr5={:5f}, user_ndcg10={:5f}, user_hr10={:5f}, user_ndcg20={:5f}, user_hr20={:5f}".
          format(train_epoch, train_step, overall_auc, user_auc, overall_logloss,
                 user_ndcg5, user_hr5, user_ndcg10, user_hr10, user_ndcg20, user_hr20))
    with open(os.path.join(args.checkpoint_dir, "log_ood.txt"), "a") as writer:
        print("train_epoch={}, train_step={}, overall_auc={:5f}, user_auc={:5f}, overall_logloss={:5f}, "
              "user_ndcg5={:5f}, user_hr5={:5f}, user_ndcg10={:5f}, user_hr10={:5f}, user_ndcg20={:5f}, user_hr20={:5f}".
              format(train_epoch, train_step, overall_auc, user_auc, overall_logloss,
                     user_ndcg5, user_hr5, user_ndcg10, user_hr10, user_ndcg20, user_hr20), file=writer)

    return overall_auc, user_auc, overall_logloss, user_ndcg5, user_hr5, user_ndcg10, user_hr10, user_ndcg20, user_hr20


def train(train_dataset, model_obj, device, args, pred_dataloader):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model_obj.parameters(),
        lr=float(args.learning_rate)
    )
    model_obj.train()
    model_obj.to(device)

    # load base param
    model_param = model_obj.state_dict()  # model to be trained
    base_model = torch.load(args.base_model_path)  # base model with seed 0
    base_param = base_model.state_dict()
    for name, param in base_param.items():
        if "_classifier" not in name:
            model_param[name] = base_param[name]

    if args.model == "graft_din":
        base_param_name = ["_classifier.net.0.weight", "_classifier.net.0.bias", "_classifier.net.2.weight", "_classifier.net.2.bias", "_classifier.net.4.weight", "_classifier.net.4.bias"]
        model_obj_param_name = ["_classifier._net_module_0.weight", "_classifier._net_module_0.bias", "_classifier._net_module_1.weight", "_classifier._net_module_1.bias", "_classifier._net_module_2.weight", "_classifier._net_module_2.bias"]
    elif args.model == "graft_gru4rec" or args.model == "graft_sasrec":
        base_param_name = ["_classifier.net.0.weight", "_classifier.net.0.bias", "_classifier.net.2.weight", "_classifier.net.2.bias"]
        model_obj_param_name = ["_classifier._net_module_0.weight", "_classifier._net_module_0.bias", "_classifier._net_module_1.weight", "_classifier._net_module_1.bias"]

    for key in range(len(base_param_name)):
        model_param[model_obj_param_name[key]] = base_param[base_param_name[key]]

    model_obj.load_state_dict(model_param)

    graft_model = torch.load(args.graft_model_path)  # graft model with seed 1
    graft_param = graft_model.state_dict()

    print(model_obj)

    logger.info("Start training...")
    timer = Timer()
    log_every = 200
    max_step = 0
    best_auc = 0
    best_auc_ckpt_path = os.path.join(args.checkpoint_dir, "best_auc" + ".pkl")
    for epoch in range(1, args.max_epoch + 1):
        for step, batch_data in enumerate(train_dataset, 1):
            logits = model_obj({
                key: value.to(device)
                for key, value in batch_data.items()
                if key not in {consts.FIELD_USER_ID, consts.FIELD_LABEL, consts.FIELD_TRIGGER_SEQUENCE}
            }, graft_param)

            loss = criterion(logits, batch_data[consts.FIELD_LABEL].view(-1, 1).to(device))
            pred, y = torch.sigmoid(logits), batch_data[consts.FIELD_LABEL].view(-1, 1)
            overall_auc, _, _, _ = new_metrics.calculate_overall_auc(np.array(pred.detach().cpu()), np.array(y))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % log_every == 0:
                logger.info(
                    "epoch={}, step={}, loss={:5f}, overall_auc={:5f}, speed={:2f} steps/s".format(
                        epoch, step, float(loss.item()), overall_auc, log_every / timer.tick(False)
                    )
                )
            max_step = step

        pred_overall_auc, pred_user_auc, pred_overall_logloss, pred_user_ndcg5, pred_user_hr5, \
        pred_user_ndcg10, pred_user_hr10, pred_user_ndcg20, pred_user_hr20 = predict(
            predict_dataset=pred_dataloader,
            model_obj=model_obj,
            device=device,
            args=args,
            train_epoch=epoch,
            train_step=epoch * max_step,
            graft_param=graft_param
        )
        logger.info("dump checkpoint for epoch {}".format(epoch))
        model_obj.train()
        if pred_overall_auc > best_auc:
            best_auc = pred_overall_auc
            torch.save(model_obj, best_auc_ckpt_path)

        # test_model = torch.load(best_auc_ckpt_path)
        # pred_overall_auc, pred_user_auc, pred_overall_logloss, pred_user_ndcg5, pred_user_hr5, \
        #     pred_user_ndcg10, pred_user_hr10, pred_user_ndcg20, pred_user_hr20 = predict(
        #     predict_dataset=pred_dataloader,
        #     model_obj=test_model,
        #     device=device,
        #     args=args,
        #     train_epoch=epoch,
        #     train_step=epoch * max_step,
        #     graft_param=graft_param
        # )


def main_worker(_):
    args = parse_args()
    ap.print_arguments(args)

    # args.checkpoint_dir = os.path.join(args.checkpoint_dir, "base")
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    model_meta = model.get_model_meta(args.model)  # type: model.ModelMeta

    model_conf, raw_model_conf = ap.parse_arch_config_from_args(model_meta, args)  # type: dict

    # Construct model
    model_obj = model_meta.model_builder(model_conf=model_conf)  # type: torch.nn.module
    print("=" * 100)
    for name, parms in model_obj.named_parameters():
        print(name)
    print("=" * 100)
    device = env.get_device()
    worker_id = worker_count = 8
    train_file, test_file = args.dataset.split(',')

    args.num_loading_workers = 1
    # Setup up data loader
    # train_dataloader = sequence_dataloader.SequenceDataLoader(
    train_dataloader = sequence_dataloader.MetaSequenceDataLoader(
        table_name=train_file,
        slice_id=0,
        slice_count=args.num_loading_workers,
        is_train=True
    )
    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataloader,
        batch_size=args.batch_size,
        num_workers=args.num_loading_workers,
        pin_memory=True,
        collate_fn=train_dataloader.batchify,
        drop_last=False,
        # shuffle=True
    )

    # Setup up data loader
    pred_dataloader = sequence_dataloader.MetaSequenceDataLoader(
        table_name=test_file,
        slice_id=args.num_loading_workers * worker_id,
        slice_count=args.num_loading_workers * worker_count,
        is_train=False
    )
    pred_dataloader = torch.utils.data.DataLoader(
        dataset=pred_dataloader,
        batch_size=args.batch_size,
        num_workers=args.num_loading_workers,
        pin_memory=True,
        collate_fn=pred_dataloader.batchify,
        drop_last=False,
    )


    train(
        train_dataset=train_dataloader,
        model_obj=model_obj,
        device=device,
        args=args,
        pred_dataloader=pred_dataloader
    )


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    mp.spawn(main_worker, nprocs=1)

