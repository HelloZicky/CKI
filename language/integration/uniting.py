"""Finetuning the library models for sequence classification on GLUE."""

import dataclasses
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, List
import torch

import numpy as np

import transformers
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, EvalPrediction, GPT2LMHeadModel, \
    RobertaForSequenceClassification
from transformers import GlueDataTrainingArguments as DataTrainingArguments
from transformers import TrainingArguments, set_seed
from src.gpt_trainer import gptTrainer

from tools.hf_argparser import HfArgumentParser
from src.dataset import FewShotDataset
from src.models import BertForPromptFinetuning, RobertaForPromptFinetuning, resize_token_type_embeddings
from src.trainer import Trainer
from src.processors import processors_mapping, num_labels_mapping, output_modes_mapping, compute_metrics_mapping, \
    bound_mapping
from src.gptdataset import gptDataset

from filelock import FileLock
from datetime import datetime

from copy import deepcopy
from tqdm import tqdm
import json

from src.methods import *
from tools.env import *

logger = logging.getLogger(__name__)

os.environ["WANDB_DISABLED"] = "true"


# print (compute_metrics_mapping['telephone_letters'])
# exit(0)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )

    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    # Few-shot type
    #   - finetune: standard fine-tuning
    #   - prompt: prompt-based fine-tuning
    #   - prompt-demo: prompt-based fine-tuning with demonstrations
    few_shot_type: str = field(
        default='prompt-demo',
        metadata={"help": "Few-shot learning model type. Choice: finetune, prompt, autoregressive"}
    )

    # Only for BERT-type model
    random_segment: bool = field(
        default=False,
        metadata={"help": "Whether to reinitialize the token type embeddings (only for BERT)."}
    )

    use_lm_head: int = field(
        default=1,
        metadata={"help": "0/1: Whether to use lm head or use a simple linear classifier."}
    )

    log_file_store: str = field(
        default='prompt-demo',
        metadata={"help": "File to log results"}
    )

    use_CLS_linearhead: int = field(
        default=0,
        metadata={"help": "0/1: Whether to use [CLS] or the mask representation."}
    )

    l1_reg: float = field(
        default=0.,
        metadata={"help": "Apply l1 regularization on the model parameters!"}
    )

    # for uniting
    graft_model_path: str = field(
        default=None,
        metadata={"help": "Path to finetuned model for uniting"}
    )

    model_type: str = field(
        default="roberta", metadata={"help": "model type of the graft model"}
    )


@dataclass
class DynamicDataTrainingArguments(DataTrainingArguments):
    """
    Arguments for dynamic training.
    """
    num_k: Optional[int] = field(
        default=16,
        metadata={"help": "Number of training instances per class"}
    )

    num_sample: Optional[int] = field(
        default=16,
        metadata={"help": "Number of samples (for inference) in fine-tuning with demonstrations"}
    )

    num_demo: Optional[int] = field(
        default=1,
        metadata={"help": "Number of demonstrations from each class"}
    )

    auto_demo: bool = field(
        default=True,
        metadata={"help": "Automatically generate template for using demonstrations"}
    )

    # For prompting
    template: str = field(
        default=None,
        metadata={"help": "Template"}
    )

    mapping: str = field(
        default=None,
        metadata={"help": "Label word mapping"}
    )

    template_path: str = field(
        default=None,
        metadata={
            "help": "Path to a txt file that stores all the templates, one per line. Do not set this when prompt_path is used"}
    )

    mapping_path: str = field(
        default=None,
        metadata={
            "help": "Path to a txt file that stores all the label word mappings, one per line. Do not set this when prompt_path is used"}
    )

    prompt_path: str = field(
        default=None,
        metadata={"help": "Path to a txt file that stores all the prompts (templates and mappings), one per line"}
    )

    template_id: int = field(
        default=None,
        metadata={"help": "Template id if using template_path"}
    )

    mapping_id: int = field(
        default=None,
        metadata={"help": "Mapping id if using template_path"}
    )

    prompt_id: int = field(
        default=None,
        metadata={"help": "Prompt id if using prompt_path"}
    )

    top_n_template: int = field(
        default=None,
        metadata={"help": "Use top-n template in the template path"}
    )

    # For logging
    tag: str = field(
        default='',
        metadata={"help": "Set the tag and find the result easier in the log."}
    )

    # For filtering when using demonstrations
    demo_filter: bool = field(
        default=False,
        metadata={"help": "Only use similar instances in demonstrations"}
    )

    demo_filter_rate: float = field(
        default=0.5,
        metadata={"help": "Only use top-x\% similar instances in demonstrations"}
    )

    demo_filter_model: str = field(
        default=None,
        metadata={
            "help": "Model name for demonstration filter embeddings. Will load embeddings based on the model name."}
    )

    debug_mode: bool = field(
        default=False,
        metadata={"help": "Debug mode"}
    )

    # For max length
    double_demo: bool = field(
        default=False,
        metadata={"help": "Use double length for using demonstrations"}
    )

    first_sent_limit: int = field(
        default=None,
        metadata={"help": "Limit the length of the first sentence (i.e., sent_0)"}
    )

    other_sent_limit: int = field(
        default=None,
        metadata={"help": "Limit the length of sentences other than the first sentence"}
    )

    use_full_length: bool = field(
        default=None,
        metadata={"help": "Use the full length (512)"}
    )

    max_length_per_example: int = field(
        default=None,
        metadata={"help": "Max length per example in gpt experiments on gpt!"}
    )

    # Arguments for gpt3 in-context experiments: not necessary for our experiments!
    gpt3_in_context_head: bool = field(
        default=False,
        metadata={"help": "GPT-3's in-context learning (context at the beginning)"}
    )

    gpt3_in_context_tail: bool = field(
        default=False,
        metadata={"help": "GPT-3's in-context learning (context at the end)"}
    )

    gpt3_in_context_num: int = field(
        default=32,
        metadata={"help": "Number of context examples"}
    )

    truncate_head: bool = field(
        default=False,
        metadata={"help": "When exceeding the maximum length, truncate the head instead of the tail."}
    )

    # Do not set up the following fields. They are set up automatically.
    prompt: bool = field(
        default=False,
        metadata={"help": "Whether to use prompt-based fine-tuning"}
    )
    template_list: List = field(
        default=None,
        metadata={"help": "(DO NOT List of templates (only initialized after the program starts."}
    )

    autoregressive: bool = field(
        default=False,
        metadata={"help": "Whether to use GPT2 fine-tuning"}
    )


@dataclass
class DynamicTrainingArguments(TrainingArguments):
    # For ensemble
    array_id: int = field(
        default=-1,
        metadata={"help": "Array ID (contains seed and hyper-paramter search) to idenfity the model"}
    )

    model_id: int = field(
        default=-1,
        metadata={"help": "Model ID (contains template information) to identify the model"}
    )

    save_logit: bool = field(
        default=False,
        metadata={"help": "Save test file logit with name $TASK-$MODEL_ID-$ARRAY_ID.npy"}
    )

    save_logit_dir: str = field(
        default=None,
        metadata={"help": "Where to save the prediction result"}
    )

    # Regularization
    fix_layers: int = field(
        default=0,
        metadata={"help": "Fix bottom-n layers when optimizing"}
    )

    fix_embeddings: bool = field(
        default=False,
        metadata={"help": "Fix embeddings when optimizing"}
    )

    fix_head: bool = field(
        default=False,
        metadata={"help": "Fix lm head when optimizing"}
    )

    # Training
    save_at_last: bool = field(
        default=False,
        metadata={"help": "Instead of saving the best (dev performance) checkpoint, save the last checkpoint"}
    )

    # Turn off train/test
    no_train: bool = field(
        default=False,
        metadata={"help": "No training"}
    )
    no_predict: bool = field(
        default=False,
        metadata={"help": "No test"}
    )

    optimizer: str = field(
        default='AdamW',
        metadata={"help": "AdamW/SGD?"}
    )

    train_head: int = field(
        default=0,
        metadata={"help": "0/1: Whether to train the head."}
    )

    save_every_ckpt: int = field(
        default=0,
        metadata={"help": "0/1: Whether to save ckpts at regular intervals."}
    )

    train_bias_only: bool = field(
        default=False,
        metadata={"help": "Whether to run bitfit!"}
    )

    # for uniting
    uniting_model_ckpt: str = field(
        default=None,
        metadata={"help": "path to the trained uniting model for uniting"}
    )

    uniting_learning_rate: float = field(
        default=0.001,
        metadata={"help": "learning rate for model uniting training"}
    )

    uniting_epochs: int = field(
        default=10,
        metadata={"help": "uniting training epochs"}
    )

    train_uniting_model: bool = field(
        default=True,
        metadata={"help": "whether to train the uniting model or not"}
    )

    method: str = field(
        default="averaging",
        metadata={"help": "method for model merging"}
    )



def main():
    parser = HfArgumentParser((ModelArguments, DynamicDataTrainingArguments, DynamicTrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if training_args.no_train:
        training_args.do_train = False
    if training_args.no_predict:
        training_args.do_predict = False

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )

    # Check save path
    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(f"Output directory ({training_args.output_dir}) already exists.")

    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    # with open(model_args.log_file_store, 'a') as f:
    #     f.write("training args: \n" + str(training_args) + "\n")
    #     f.write("model args: \n" + str(model_args) + "\n")
    #     f.write("data args: \n" + str(data_args) + "\n")
    #
    # return


    # Set seed
    set_seed(training_args.seed)

    try:
        num_labels = num_labels_mapping[data_args.task_name]
        output_mode = output_modes_mapping[data_args.task_name]
        logger.info(
            "Task name: {}, number of labels: {}, output mode: {}".format(data_args.task_name, num_labels, output_mode))
    except KeyError:
        raise ValueError("Task not found: %s" % (data_args.task_name))

    special_tokens = []

    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        additional_special_tokens=special_tokens,
        cache_dir=model_args.cache_dir,
    )

    data_cache_dir = data_args.data_dir

    if not os.path.exists(data_cache_dir):
        os.mkdir(data_cache_dir)

    log_file_store = model_args.log_file_store
    if not os.path.exists(log_file_store):
        os.mkdir(log_file_store)

    log_file_store = log_file_store + "/log"

    dataset_class = FewShotDataset

    # Get our special datasets.
    train_dataset = (
        dataset_class(data_args, tokenizer=tokenizer, cache_dir=data_cache_dir, mode="train",
                      use_demo=("demo" in model_args.few_shot_type))
    )

    eval_dataset = (
        dataset_class(data_args, tokenizer=tokenizer, cache_dir=data_cache_dir, mode="dev",
                      use_demo=("demo" in model_args.few_shot_type))
        if training_args.do_eval
        else None
    )

    # test_dataset = (
    #     dataset_class(data_args, tokenizer=tokenizer, cache_dir=data_cache_dir, mode="test",
    #                   use_demo=("demo" in model_args.few_shot_type))
    #     if training_args.do_predict
    #     else None
    # )
    test_dataset = (
        dataset_class(data_args, tokenizer=tokenizer, cache_dir=data_cache_dir, mode="dev",
                      use_demo=("demo" in model_args.few_shot_type))
        if training_args.do_predict
        else None
    )

    set_seed(training_args.seed)

    # Create config
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
    )

    device = get_device()

    model_fn = AutoModelForSequenceClassification

    model = model_fn.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )

    model = RobertaForSequenceClassification(config=config)
    # for n, p in model.roberta.named_parameters():
    #     p.requires_grad_(False)
    model.to(device)

    # load model for grafting
    graft_model = model_fn.from_pretrained(
        model_args.graft_model_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )
    graft_model.to(device)

    graft_params = graft_model.classifier.state_dict()

    # For BERT, increase the size of the segment (token type) embeddings
    if config.model_type == 'bert':
        model.resize_token_embeddings(len(tokenizer))
        resize_token_type_embeddings(model, new_num_types=10, random_segment=model_args.random_segment)

    # Pass dataset and argument information to the model
    if data_args.prompt:
        model.label_word_list = torch.tensor(train_dataset.label_word_list).long().cuda()
    if output_modes_mapping[data_args.task_name] == 'regression':
        # lower / upper bounds
        model.lb, model.ub = bound_mapping[data_args.task_name]
    model.model_args = model_args
    model.data_args = data_args
    model.tokenizer = tokenizer

    model.initial_parameters_copy = [p.detach().clone() for p in model.parameters()]

    # Build metric
    def build_compute_metrics_fn(task_name: str) -> Callable[[EvalPrediction], Dict]:
        def compute_metrics_fn(p: EvalPrediction):
            # Note: the eval dataloader is sequential, so the examples are in order.
            # We average the logits over each sample for using demonstrations.
            predictions = p.predictions
            num_logits = predictions.shape[-1]
            logits = predictions.reshape([eval_dataset.num_sample, -1, num_logits])
            logits = logits.mean(axis=0)

            if num_logits == 1:
                preds = np.squeeze(logits)
            else:
                preds = np.argmax(logits, axis=1)

            # Just for sanity, assert label ids are the same.
            label_ids = p.label_ids.reshape([eval_dataset.num_sample, -1])
            label_ids_avg = label_ids.mean(axis=0)
            label_ids_avg = label_ids_avg.astype(p.label_ids.dtype)
            assert (label_ids_avg - label_ids[0]).mean() < 1e-2
            label_ids = label_ids[0]

            if 'telephone' in task_name:
                return compute_metrics_mapping[task_name]('mnli', preds, label_ids)
            if 'anli' in task_name:
                return compute_metrics_mapping[task_name]('mnli', preds, label_ids)
            if 'imdb' in task_name:
                return compute_metrics_mapping[task_name]('sst-2', preds, label_ids)
            return compute_metrics_mapping[task_name](task_name, preds, label_ids)

        return compute_metrics_fn

    if training_args.method == "averaging":
        model = averaging_models(model, graft_params, model_args, device)
    elif training_args.method == "uniting":
        trainer_class = Trainer
        trainer = trainer_class(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=build_compute_metrics_fn(data_args.task_name)
        )
        model = uniting(model, graft_params, config, model_args, training_args, train_dataset, eval_dataset, trainer, device, logger)
    elif training_args.method == "unimodel":
        pass
    elif training_args.method == "pruning":
        trainer_class = Trainer
        trainer = trainer_class(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=build_compute_metrics_fn(data_args.task_name)
        )
        model = pruning(model, config, model_args, training_args, train_dataset, eval_dataset, trainer, device, logger)
    elif training_args.method == "ensemble":
        trainer_class = Trainer
        trainer = trainer_class(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=build_compute_metrics_fn(data_args.task_name)
        )
        model = ensemble(model, graft_model, config, model_args, training_args, train_dataset, eval_dataset, trainer, device, logger)
    elif training_args.method == "finetune":
        trainer_class = Trainer
        trainer = trainer_class(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=build_compute_metrics_fn(data_args.task_name)
        )
        trainer.train()
        trainer.save_model(training_args.uniting_model_ckpt)
        config.save_pretrained(training_args.uniting_model_ckpt)
        tokenizer.save_pretrained(training_args.uniting_model_ckpt)
        torch.save(model_args, os.path.join(training_args.uniting_model_ckpt, "model_args.bin"))
        torch.save(data_args, os.path.join(training_args.uniting_model_ckpt, "data_args.bin"))

        # reload the finetuned model
        model = model_fn.from_pretrained(training_args.uniting_model_ckpt)


    # Initialize our Trainer
    trainer_class = Trainer

    # Initialize our Trainer
    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=build_compute_metrics_fn(data_args.task_name)
    )

    final_result = {
        'time': str(datetime.today()),
    }

    eval_results = {}
    if training_args.do_eval:
        logger.info("*** Validate ***")

        eval_datasets = [eval_dataset]

        for eval_dataset in eval_datasets:
            trainer.compute_metrics = build_compute_metrics_fn(eval_dataset.args.task_name)
            output = trainer.evaluate(eval_dataset=eval_dataset)
            eval_result = output.metrics

            output_eval_file = os.path.join(
                training_args.output_dir, f"eval_results_{eval_dataset.args.task_name}.txt"
            )

            with open(output_eval_file, "a") as writer:
                logger.info("***** Eval results {} *****".format(eval_dataset.args.task_name))
                for key, value in eval_result.items():
                    logger.info("  %s = %s", key, value)
                    writer.write("epoch=%d, %s=%s\n" % (training_args.uniting_epochs, key, value))
                    final_result[eval_dataset.args.task_name + '_dev_' + key] = value
            eval_results.update(eval_result)

    test_results = {}
    if training_args.do_predict:
        logging.info("*** Test ***")
        test_datasets = [test_dataset]
        if data_args.task_name == "mnli":
            mnli_mm_data_args = dataclasses.replace(data_args, task_name="mnli-mm")
            test_datasets.append(
                FewShotDataset(mnli_mm_data_args, tokenizer=tokenizer, mode="test",
                               use_demo=('demo' in model_args.few_shot_type))
            )

        for test_dataset in test_datasets:
            trainer.compute_metrics = build_compute_metrics_fn(test_dataset.args.task_name)
            output = trainer.evaluate(eval_dataset=test_dataset)
            test_result = output.metrics

            output_test_file = os.path.join(
                training_args.output_dir, f"test_results_{test_dataset.args.task_name}.txt"
            )

            with open(output_test_file, "a") as writer:
                logger.info("***** Test results {} *****".format(test_dataset.args.task_name))
                for key, value in test_result.items():
                    logger.info("  %s = %s", key, value)
                    writer.write("%s = %s\n" % (key, value))
                    final_result[test_dataset.args.task_name + '_test_' + key] = value

                if training_args.save_logit:
                    predictions = output.predictions
                    num_logits = predictions.shape[-1]
                    logits = predictions.reshape([test_dataset.num_sample, -1, num_logits]).mean(axis=0)
                    np.save(os.path.join(training_args.save_logit_dir,
                                         "{}-{}-{}.npy".format(test_dataset.task_name, training_args.model_id,
                                                               training_args.array_id)), logits)

            test_results.update(test_result)

    with FileLock(log_file_store + '.lock'):
        # 'log_noembed_SGD_linearhead.lock'):
        with open(log_file_store, 'a') as f:
            final_result.update(vars(model_args))
            final_result.update(vars(training_args))
            final_result.update(vars(data_args))
            if 'evaluation_strategy' in final_result:
                final_result.pop('evaluation_strategy')
            f.write(str(final_result) + '\n')

    return eval_results


if __name__ == "__main__":
    main()
