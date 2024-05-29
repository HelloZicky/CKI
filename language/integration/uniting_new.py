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
from datasets import load_dataset, load_metric, load_from_disk
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, EvalPrediction, GPT2LMHeadModel, \
    Trainer, RobertaForSequenceClassification
from transformers import GlueDataTrainingArguments as DataTrainingArguments
from transformers import TrainingArguments, set_seed
from src.gpt_trainer import gptTrainer

from tools.hf_argparser import HfArgumentParser
from src.dataset import FewShotDataset
from src.models import BertForPromptFinetuning, RobertaForPromptFinetuning, resize_token_type_embeddings

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
    base_model_path: str = field(
        default=None,
        metadata={"help": "Path to finetuned model for uniting"}
    )

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

    finetune_model_ckpt: str = field(
        default=None,
        metadata={"help": "path to the finetuned model"}
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

    GLUE_TASKS = ["cola", "mnli", "mnli-mm", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"]

    task = data_args.task_name
    batch_size = training_args.per_device_train_batch_size
    data_dir = data_args.data_dir
    device = get_device()

    log_file_store = training_args.output_dir
    if not os.path.exists(log_file_store):
        os.mkdir(log_file_store)

    log_file_store = log_file_store + "/log"

    actual_task = "mnli" if task == "mnli-mm" else task
    dataset = load_from_disk(data_dir)
    metric = load_metric('glue', actual_task)

    # Set seed
    set_seed(training_args.seed)

    task_to_keys = {
        "cola": ("sentence", None),
        "mnli": ("premise", "hypothesis"),
        "mnli-mm": ("premise", "hypothesis"),
        "mrpc": ("sentence1", "sentence2"),
        "qnli": ("question", "sentence"),
        "qqp": ("question1", "question2"),
        "rte": ("sentence1", "sentence2"),
        "sst2": ("sentence", None),
        "stsb": ("sentence1", "sentence2"),
        "wnli": ("sentence1", "sentence2"),
    }

    num_labels = 3 if task.startswith("mnli") else 1 if task == "stsb" else 2

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, use_fast=True)
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=task,
        cache_dir=model_args.cache_dir,
    )

    # preprocess data
    sentence1_key, sentence2_key = task_to_keys[task]
    if sentence2_key is None:
        print(f"Sentence: {dataset['train'][0][sentence1_key]}")
    else:
        print(f"Sentence 1: {dataset['train'][0][sentence1_key]}")
        print(f"Sentence 2: {dataset['train'][0][sentence2_key]}")

    def preprocess_function(examples):
        if sentence2_key is None:
            return tokenizer(examples[sentence1_key], truncation=True)
        return tokenizer(examples[sentence1_key], examples[sentence2_key], truncation=True)

    encoded_dataset = dataset.map(preprocess_function, batched=True)

    # initial pretrain model
    model_fn = AutoModelForSequenceClassification
    model = model_fn.from_pretrained(model_args.model_name_or_path, num_labels=num_labels)
    # freeze layers except for classifier
    if config.model_type == "roberta":
        for n, p in model.roberta.named_parameters():
            p.requires_grad_(False)
    elif config.model_type == "bert":
        for n, p in model.bert.named_parameters():
            p.requires_grad_(False)

    # # initial the whole model
    # model_fn = RobertaForSequenceClassification
    # model = model_fn(config=config)

    metric_name = "pearson" if task == "stsb" else "matthews_correlation" if task == "cola" else "accuracy"
    model_name = model_args.model_type

    args = TrainingArguments(
        training_args.finetune_model_ckpt,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=training_args.uniting_learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=training_args.num_train_epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model=metric_name,
        push_to_hub=False,
    )

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        if task != "stsb":
            predictions = np.argmax(predictions, axis=1)
        else:
            predictions = predictions[:, 0]
        return metric.compute(predictions=predictions, references=labels)

    validation_key = "validation_mismatched" if task == "mnli-mm" else "validation_matched" if task == "mnli" else "validation"
    trainer = Trainer(
        model,
        args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset[validation_key],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )


    if training_args.method == "finetune":
        trainer.train()
        trainer.save_model(training_args.finetune_model_ckpt)
        config.save_pretrained(training_args.finetune_model_ckpt)
        tokenizer.save_pretrained(training_args.finetune_model_ckpt)
        torch.save(model_args, os.path.join(training_args.finetune_model_ckpt, "model_args.bin"))
        torch.save(data_args, os.path.join(training_args.finetune_model_ckpt, "data_args.bin"))

        # reload the finetuned model
        model = model_fn.from_pretrained(training_args.finetune_model_ckpt)
    else:
        model = model_fn.from_pretrained(model_args.base_model_path)
        model.to(device)

        graft_model = model_fn.from_pretrained(model_args.graft_model_path)
        graft_model.to(device)
        graft_params = copy.deepcopy(graft_model.classifier.state_dict())
        if training_args.method == "averaging":
            model = averaging_models(model, graft_params, model_args, device)
        elif training_args.method == "uniting":
            model = uniting(model, graft_params, config, model_args, training_args, trainer, device, logger, task)
        elif training_args.method == "pruning":
            model = pruning(model, graft_params, config, model_args, training_args, trainer, device, logger, task)
        elif training_args.method == "ensemble":
            model = ensemble(model, graft_model, graft_params, config, model_args, training_args, trainer, device, logger, task)
        elif training_args.method == "unimodel":
            pass

    model.to(device)
    trainer.model = model
    final_result = {
        'time': str(datetime.today()),
    }

    eval_results = {}
    logger.info("*** Validate ***")
    eval_output = trainer.evaluate()

    output_eval_file = os.path.join(
        training_args.output_dir, f"eval_results_{task}.txt"
    )

    with open(output_eval_file, "a") as writer:
        logger.info("***** Eval results {} *****".format(task))
        writer.write("epoch={}, eval_loss={:5f}, eval_accuracy={:5f}"
                     .format(training_args.num_train_epochs, eval_output["eval_loss"], eval_output["eval_accuracy"]))
        for key, value in eval_output.items():
            final_result[task + '_dev_' + key] = value
            logger.info("  %s = %s", key, value)
    eval_results.update(eval_output)

    with open(log_file_store, 'a') as f:
        final_result.update(vars(model_args))
        final_result.update(vars(training_args))
        final_result.update(vars(data_args))
        if 'evaluation_strategy' in final_result:
            final_result.pop('evaluation_strategy')
        f.write(str(final_result) + '\n')



if __name__ == "__main__":
    main()
