import copy
import os
from typing import Callable, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.nn import MSELoss, CrossEntropyLoss, BCEWithLogitsLoss
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler

from src.models import UnitingModelForNLP, PruningModelForNLP, EnsembleModelForNLP
from src.processors import processors_mapping, num_labels_mapping, output_modes_mapping, compute_metrics_mapping, \
    bound_mapping

from tqdm import tqdm, trange
from transformers import EvalPrediction


def averaging_models(model, graft_param, model_args, device):
    base_param = model.classifier.state_dict()

    if model_args.model_type == "roberta":
        base_param_name = ["dense.weight", "out_proj.weight"]
    elif model_args.model_type == "bert":
        base_param_name = ["weight"]

    for i in range(len(base_param_name)):
        base_param[base_param_name[i]] = (graft_param[base_param_name[i]] + base_param[base_param_name[i]]) / 2

    model.classifier.load_state_dict(base_param)
    return model

def uniting(model, graft_param, config, model_args, training_args, trainer, device, logger, task):
    base_param = copy.deepcopy(model.classifier.state_dict())

    if model_args.model_type == "roberta":
        base_param_name = ["dense.weight", "out_proj.weight"]

    if training_args.uniting_model_ckpt and not training_args.train_uniting_model:
        uniting_model = torch.load(training_args.uniting_model_ckpt)
        return uniting_model.load_param(model, graft_param)
    else:
        uniting_model = UnitingModelForNLP(config, model_args.model_type)
        uniting_model.to(device)
        uniting_model.train()

        train_dataloader = trainer.get_train_dataloader()
        optimizer = torch.optim.Adam(
            uniting_model.parameters(),
            lr=float(training_args.uniting_learning_rate)
        )

        # training_args.uniting_epochs = 1
        log_every = 50
        best_res = 0
        best_uniting_ckpt_path = os.path.join(training_args.uniting_model_ckpt, "best_res" + ".pkl")
        for epoch in range(training_args.uniting_epochs):
            logger.info("***Training Epoch {}***".format(epoch + 1))
            epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=True)
            for step, inputs in enumerate(epoch_iterator):
                loss = None
                input_ids = inputs["input_ids"].to(device)
                labels = inputs["labels"].to(device)
                attention_mask = inputs["attention_mask"].to(device)

                if config.model_type == "roberta":
                    # with torch.no_grad():
                    #     roberta_output = model.roberta(input_ids=input_ids, attention_mask=attention_mask)
                    #     sequence_output = roberta_output[0]

                    roberta_output = model.roberta(input_ids=input_ids, attention_mask=attention_mask)
                    sequence_output = roberta_output[0]
                elif config.model_type == "bert":
                    bert_output = model.bert(input_ids=input_ids, attention_mask=attention_mask)
                    pooled_output = bert_output[1]

                    sequence_output = model.dropout(pooled_output)

                logits = uniting_model(sequence_output, base_param, graft_param)

                if labels is not None:
                    if config.problem_type is None:
                        if config.num_labels == 1:
                            config.problem_type = "regression"
                        elif config.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                            config.problem_type = "single_label_classification"
                        else:
                            config.problem_type = "multi_label_classification"

                    if config.problem_type == "regression":
                        loss_fct = MSELoss()
                        if config.num_labels == 1:
                            loss = loss_fct(logits.squeeze(), labels.squeeze())
                        else:
                            loss = loss_fct(logits, labels)
                    elif config.problem_type == "single_label_classification":
                        loss_fct = CrossEntropyLoss()
                        loss = loss_fct(logits.view(-1, config.num_labels), labels.view(-1))
                    elif config.problem_type == "multi_label_classification":
                        loss_fct = BCEWithLogitsLoss()
                        loss = loss_fct(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if step % log_every == 0:
                    logger.info(
                        "epoch={}, step={}, loss={:5f}".format(epoch+1, step, float(loss.item()))
                    )

            # validate
            logger.info("*** Validate ***")
            uniting_model.eval()
            model = uniting_model.load_param(model, base_param, graft_param)
            trainer.model = model


            eval_output = trainer.evaluate()
            output_eval_file = os.path.join(
                training_args.output_dir, f"eval_results_{task}.txt"
            )

            with open(output_eval_file, "a") as writer:
                logger.info("*****Epoch {}: eval results {} *****".format(epoch + 1, task))
                writer.write("epoch={}, eval_loss={:5f}, eval_accuracy={:5f}\n"
                             .format(epoch + 1, eval_output["eval_loss"], eval_output["eval_accuracy"]))
                for key, value in eval_output.items():
                    logger.info("  %s = %s", key, value)

            uniting_model.train()
            if eval_output["eval_accuracy"] > best_res:
                best_res = eval_output["eval_accuracy"]
                torch.save(uniting_model, best_uniting_ckpt_path)

        uniting_model = torch.load(best_uniting_ckpt_path)

        return uniting_model.load_param(model, base_param, graft_param)

def pruning(model, graft_param, config, model_args, training_args, trainer, device, logger, task):
    base_param = copy.deepcopy(model.classifier.state_dict())

    if model_args.model_type == "roberta":
        base_param_name = ["dense.weight", "out_proj.weight"]

    if training_args.uniting_model_ckpt and not training_args.train_uniting_model:
        uniting_model = torch.load(training_args.uniting_model_ckpt)
        model.classifier = uniting_model
        return model
    else:
        out_proj_weight_mask = model.classifier.weight.data > 0.02
        out_proj_bias_mask = model.classifier.bias.data > 0.02
        model.classifier.weight.data *= out_proj_weight_mask.float()
        model.classifier.bias.data *= out_proj_bias_mask.float()
        # uniting_model.to(device)
        # uniting_model.train()
        #
        # train_dataloader = trainer.get_train_dataloader()
        # optimizer = torch.optim.Adam(
        #     uniting_model.parameters(),
        #     lr=float(training_args.uniting_learning_rate)
        # )
        #
        # # training_args.uniting_epochs = 1
        # log_every = 50
        # best_res = 0
        # best_uniting_ckpt_path = os.path.join(training_args.uniting_model_ckpt, "best_res" + ".pkl")
        # for epoch in range(training_args.uniting_epochs):
        #     logger.info("***Training Epoch {}***".format(epoch + 1))
        #     epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=True)
        #     for step, inputs in enumerate(epoch_iterator):
        #         loss = None
        #         input_ids = inputs["input_ids"].to(device)
        #         labels = inputs["labels"].to(device)
        #         attention_mask = inputs["attention_mask"].to(device)
        #
        #         with torch.no_grad():
        #             roberta_output = model.roberta(input_ids=input_ids, attention_mask=attention_mask)
        #             sequence_output = roberta_output[0]
        #
        #         # roberta_output = model.roberta(input_ids=input_ids, attention_mask=attention_mask)
        #         # sequence_output = roberta_output[0]
        #
        #         logits = uniting_model(sequence_output)
        #
        #         if labels is not None:
        #             if config.problem_type is None:
        #                 if config.num_labels == 1:
        #                     config.problem_type = "regression"
        #                 elif config.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
        #                     config.problem_type = "single_label_classification"
        #                 else:
        #                     config.problem_type = "multi_label_classification"
        #
        #             if config.problem_type == "regression":
        #                 loss_fct = MSELoss()
        #                 if config.num_labels == 1:
        #                     loss = loss_fct(logits.squeeze(), labels.squeeze())
        #                 else:
        #                     loss = loss_fct(logits, labels)
        #             elif config.problem_type == "single_label_classification":
        #                 loss_fct = CrossEntropyLoss()
        #                 loss = loss_fct(logits.view(-1, config.num_labels), labels.view(-1))
        #             elif config.problem_type == "multi_label_classification":
        #                 loss_fct = BCEWithLogitsLoss()
        #                 loss = loss_fct(logits, labels)
        #
        #         optimizer.zero_grad()
        #         loss.backward()
        #         optimizer.step()
        #
        #         if step % log_every == 0:
        #             logger.info(
        #                 "epoch={}, step={}, loss={:5f}".format(epoch+1, step, float(loss.item()))
        #             )
        #
        #     # validate
        #     logger.info("*** Validate ***")
        #     trainer.model = model
        #
        #
        #     eval_output = trainer.evaluate()
        #     output_eval_file = os.path.join(
        #         training_args.output_dir, f"eval_results_{task}.txt"
        #     )
        #
        #     with open(output_eval_file, "a") as writer:
        #         logger.info("*****Epoch {}: eval results {} *****".format(epoch + 1, task))
        #         writer.write("epoch={}, eval_loss={:5f}, eval_accuracy={:5f}\n"
        #                      .format(epoch + 1, eval_output["eval_loss"], eval_output["eval_accuracy"]))
        #         for key, value in eval_output.items():
        #             logger.info("  %s = %s", key, value)
        #
        #     uniting_model.train()
        #     if eval_output["eval_accuracy"] > best_res:
        #         best_res = eval_output["eval_accuracy"]
        #         torch.save(uniting_model, best_uniting_ckpt_path)
        #
        # uniting_model = torch.load(best_uniting_ckpt_path)
        # model.classifier = uniting_model

        return model

def ensemble(model_i, model_j, graft_param, config, model_args, training_args, trainer, device, logger, task):
    base_param = copy.deepcopy(model_i.classifier.state_dict())

    if model_args.model_type == "roberta":
        base_param_name = ["dense.weight", "out_proj.weight"]

    if training_args.uniting_model_ckpt and not training_args.train_uniting_model:
        uniting_model = torch.load(training_args.uniting_model_ckpt)
        model_i.classifier = uniting_model
        return model_i
    else:
        uniting_model = EnsembleModelForNLP(model_i, model_j)
        uniting_model.to(device)
        uniting_model.train()

        train_dataloader = trainer.get_train_dataloader()
        optimizer = torch.optim.Adam(
            uniting_model.parameters(),
            lr=float(training_args.uniting_learning_rate)
        )

        # training_args.uniting_epochs = 1
        log_every = 50
        best_res = 0
        best_uniting_ckpt_path = os.path.join(training_args.uniting_model_ckpt, "best_res" + ".pkl")
        for epoch in range(training_args.uniting_epochs):
            logger.info("***Training Epoch {}***".format(epoch + 1))
            epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=True)
            for step, inputs in enumerate(epoch_iterator):
                loss = None
                input_ids = inputs["input_ids"].to(device)
                labels = inputs["labels"].to(device)
                attention_mask = inputs["attention_mask"].to(device)

                if config.model_type == "roberta":
                    # with torch.no_grad():
                    #     roberta_output = model.roberta(input_ids=input_ids, attention_mask=attention_mask)
                    #     sequence_output = roberta_output[0]

                    roberta_output = model_i.roberta(input_ids=input_ids, attention_mask=attention_mask)
                    sequence_output = roberta_output[0]
                elif config.model_type == "bert":
                    bert_output = model_i.bert(input_ids=input_ids, attention_mask=attention_mask)
                    pooled_output = bert_output[1]

                    sequence_output = model_i.dropout(pooled_output)

                logits = uniting_model(sequence_output)

                if labels is not None:
                    if config.problem_type is None:
                        if config.num_labels == 1:
                            config.problem_type = "regression"
                        elif config.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                            config.problem_type = "single_label_classification"
                        else:
                            config.problem_type = "multi_label_classification"

                    if config.problem_type == "regression":
                        loss_fct = MSELoss()
                        if config.num_labels == 1:
                            loss = loss_fct(logits.squeeze(), labels.squeeze())
                        else:
                            loss = loss_fct(logits, labels)
                    elif config.problem_type == "single_label_classification":
                        loss_fct = CrossEntropyLoss()
                        loss = loss_fct(logits.view(-1, config.num_labels), labels.view(-1))
                    elif config.problem_type == "multi_label_classification":
                        loss_fct = BCEWithLogitsLoss()
                        loss = loss_fct(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if step % log_every == 0:
                    logger.info(
                        "epoch={}, step={}, loss={:5f}".format(epoch+1, step, float(loss.item()))
                    )

            # validate
            logger.info("*** Validate ***")
            uniting_model.eval()
            model_i.classifier = uniting_model
            trainer.model = model_i


            eval_output = trainer.evaluate()
            output_eval_file = os.path.join(
                training_args.output_dir, f"eval_results_{task}.txt"
            )

            with open(output_eval_file, "a") as writer:
                logger.info("*****Epoch {}: eval results {} *****".format(epoch + 1, task))
                writer.write("epoch={}, eval_loss={:5f}, eval_accuracy={:5f}\n"
                             .format(epoch + 1, eval_output["eval_loss"], eval_output["eval_accuracy"]))
                for key, value in eval_output.items():
                    logger.info("  %s = %s", key, value)

            uniting_model.train()
            if eval_output["eval_accuracy"] > best_res:
                best_res = eval_output["eval_accuracy"]
                torch.save(uniting_model, best_uniting_ckpt_path)

        uniting_model = torch.load(best_uniting_ckpt_path)
        model_i.classifier = uniting_model

        return model_i

