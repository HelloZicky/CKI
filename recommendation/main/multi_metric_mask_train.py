# coding=utf-8
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
from util import new_metrics
from loader import multi_metric_meta_sequence_dataloader as meta_sequence_dataloader
import numpy as np
from thop import profile

from util import utils
utils.setup_seed(0)

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables", type=str, help="Kernels configuration for CNN")
    parser.add_argument("--bucket", type=str, default=None, help="Bucket name for external storage")
    parser.add_argument("--dataset", type=str, default="alipay", help="Bucket name for external storage")
    parser.add_argument("--model_mask", type=str, default="din", help="model_mask type")
    parser.add_argument("--base_model_path", type=str, default="")
    parser.add_argument("--max_steps", type=int, help="Number of iterations before stopping")
    parser.add_argument("--snapshot", type=int, help="Number of iterations to dump model")
    parser.add_argument("--checkpoint_dir", type=str, help="Path of the checkpoint path")
    parser.add_argument("--learning_rate", type=str, default=0.001)
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--max_epoch", type=int,  default=10, help="Max epoch")
    parser.add_argument("--num_loading_workers", type=int, default=4, help="Number of threads for loading")
    parser.add_argument("--model", type=str, help="model type")
    parser.add_argument("--init_checkpoint", type=str, default="", help="Path of the checkpoint path")
    parser.add_argument("--init_step", type=int, default=0, help="Path of the checkpoint path")

    parser.add_argument("--max_gradient_norm", type=float, default=0.)

    parser.add_argument("--arch_config_path", type=str, default=None, help="Path of model configs")
    parser.add_argument("--arch_config", type=str, default=None, help="base64-encoded model configs")
    parser.add_argument("--duet_model_path", type=str, default=None)

    return parser.parse_known_args()[0]


def predict(predict_dataset, model_obj, model_obj_duet, device, args, train_epoch, train_step, writer=None):

    model_obj.eval()
    model_obj.to(device)

    timer = Timer()
    log_every = 200

    pred_list = []
    y_list = []
    buffer = []
    user_id_list = []
    for step, batch_data in enumerate(predict_dataset, 1):
        _, duet_param = model_obj_duet({
            key: value.to(device)
            for key, value in batch_data.items()
            if key not in {consts.FIELD_USER_ID, consts.FIELD_LABEL}
        })
        logits = model_obj({
            key: value.to(device)
            for key, value in batch_data.items()
            if key not in {consts.FIELD_USER_ID, consts.FIELD_LABEL}
        }, duet_param)
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

def train(train_dataset, model_obj_duet, model_obj_mask, device, args, pred_dataloader):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model_obj_mask.parameters(),
        lr=float(args.learning_rate)
    )
    model_obj_mask.train()
    model_obj_duet.train()
    model_obj_duet.to(device)
    model_obj_mask.to(device)
    # print(model_obj_mask)

    logger.info("Start training...")
    timer = Timer()
    log_every = 200
    max_step = 0
    best_auc = 0
    best_auc_ckpt_path = os.path.join(args.checkpoint_dir, "best_auc" + ".pkl")

    for epoch in range(1, args.max_epoch + 1):
        for step, batch_data in enumerate(train_dataset, 1):
            _, duet_param = model_obj_duet({
                key: value.to(device)
                for key, value in batch_data.items()
                if key not in {consts.FIELD_USER_ID, consts.FIELD_LABEL}
            })
            for param in model_obj_duet.parameters():
                param.requires_grad = False
            logits = model_obj_mask({
                key: value.to(device)
                for key, value in batch_data.items()
                if key not in {consts.FIELD_USER_ID, consts.FIELD_LABEL}
            }, duet_param)
            # print("logits done")
            loss = criterion(logits, batch_data[consts.FIELD_LABEL].view(-1, 1).to(device))
            pred, y = torch.sigmoid(logits), batch_data[consts.FIELD_LABEL].view(-1, 1)
            auc, _, _, _ = new_metrics.calculate_overall_auc(np.array(pred.detach().cpu()), np.array(y))
            optimizer.zero_grad()
            loss.backward()
            for param in model_obj_duet.parameters():
                param.grad = torch.zeros_like(param)
            optimizer.step()

            if step % log_every == 0:
                logger.info(
                    "epoch={}, step={}, loss={:5f}, auc={:5f}, speed={:2f} steps/s".format(
                        epoch, step, float(loss.item()), auc, log_every / timer.tick(False)
                    )
                )
            max_step = step

        pred_overall_auc, pred_user_auc, pred_overall_logloss, pred_user_ndcg5, pred_user_hr5, \
        pred_user_ndcg10, pred_user_hr10, pred_user_ndcg20, pred_user_hr20 = predict(
            predict_dataset=pred_dataloader,
            model_obj=model_obj_mask,
            model_obj_duet=model_obj_duet,
            device=device,
            args=args,
            train_epoch=epoch,
            train_step=epoch * max_step,
        )
        logger.info("dump checkpoint for epoch {}".format(epoch))
        model_obj_mask.train()
        if pred_overall_auc > best_auc:
            best_auc = pred_overall_auc
            torch.save(model_obj_mask, best_auc_ckpt_path)

def main_worker(_):
    args = parse_args()
    ap.print_arguments(args)

    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    model_meta_mask = model.get_model_meta(args.model_mask)
    model_meta_duet = model.get_model_meta(args.model)  # type: model.ModelMeta

    model_conf_duet, raw_model_conf_duet = ap.parse_arch_config_from_args(model_meta_duet, args)  # type: dict
    model_conf_mask, raw_model_conf_mask = ap.parse_arch_config_from_args(model_meta_mask, args)
    model_obj_duet = model_meta_duet.model_builder(model_conf=model_conf_duet)  # type: torch.nn.module
    base_model = torch.load(args.base_model_path)
    base_param = base_model.state_dict()

    # load duet model
    model_obj_duet = torch.load(args.duet_model_path)

    
    model_obj_mask = model_meta_mask.model_builder(model_conf=model_conf_mask, base_param=base_param)
    # print("=" * 100)
    # for name, parms in model_obj.named_parameters():
    #     print(name)
    # print("=" * 100)
    device = env.get_device()

    worker_id = worker_count = 8
    train_file, test_file = args.dataset.split(',')

    args.num_loading_workers = 1
    train_dataloader = meta_sequence_dataloader.MetaSequenceDataLoader(
        table_name=train_file,
        slice_id=0,
        slice_count=args.num_loading_workers,
        is_train=True
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataloader,
        batch_size=args.batch_size,
        num_workers=args.num_loading_workers,
        pin_memory=True,
        collate_fn=train_dataloader.batchify,
        drop_last=False
    )

    # Setup up data loader
    pred_dataloader = meta_sequence_dataloader.MetaSequenceDataLoader(
        table_name=test_file,
        slice_id=args.num_loading_workers * worker_id,
        slice_count=args.num_loading_workers * worker_count,
        is_train=False
    )
    pred_dataloader = torch.utils.data.DataLoader(
        pred_dataloader,
        batch_size=args.batch_size,
        num_workers=args.num_loading_workers,
        pin_memory=True,
        collate_fn=pred_dataloader.batchify,
        drop_last=False
    )

    # Setup training
    train(
        train_dataset=train_dataloader,
        model_obj_duet=model_obj_duet,
        model_obj_mask=model_obj_mask,
        device=device,
        args=args,
        pred_dataloader=pred_dataloader
    )


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    mp.spawn(main_worker, nprocs=1)

