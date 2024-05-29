import os
import logging
import torch
from datasets import load_from_disk, load_metric, Dataset
from torch import softmax
from torch.nn.functional import log_softmax
from transformers import AutoModelForSequenceClassification, AutoTokenizer, set_seed
from absl import app, flags
from torch.utils.data import DataLoader
import h5py

import fisher

_LIST_GROUP_NAME = "__list__"

def set_h5_ds(ds, val):
    if not val.shape:
        # Scalar
        ds[()] = val
    else:
        ds[:] = val

def save_variables_to_hdf5(variables, filepath):
    with h5py.File(filepath, "w") as f:
        ls = f.create_group(_LIST_GROUP_NAME)
        ls.attrs["length"] = len(variables)
        for i, v in enumerate(variables):
            val = v.detach().cpu().numpy()
            ds = ls.create_dataset(str(i), val.shape, dtype=val.dtype)
            set_h5_ds(ds, val)
            # 在PyTorch中，参数没有'name'属性，所以这里使用索引i作为名称
            name = f"param_{i}"
            ds.attrs["name"] = name
            ds.attrs["trainable"] = v.requires_grad

def _batch_size(batch):
    return batch['input_ids'].size(0)

def _compute_exact_fisher_for_batch(batch, model, variables, expectation_wrt_logits=True):
    assert expectation_wrt_logits, "TODO: Handle sampling from logits."
    num_labels = model.config.num_labels

    def fisher_single_example(single_example_batch):
        model.eval()
        single_example_batch = {k: v.squeeze(0) for k, v in single_example_batch.items()}
        logits = model(**single_example_batch).logits
        log_probs = log_softmax(logits, dim=-1)
        probs = softmax(logits, dim=-1)

        sq_grads = []
        for i in range(num_labels):
            log_prob = log_probs[0, i]
            model.zero_grad()
            log_prob.backward(retain_graph=True)
            grad = [v.grad for v in variables]
            sq_grad = [probs[0, i] * (g ** 2) for g in grad]
            sq_grads.append(sq_grad)

        example_fisher = [torch.sum(torch.stack(g), dim=0) for g in zip(*sq_grads)]
        return example_fisher

    fishers = [fisher_single_example({k: v[:, i, :] for k, v in batch.items()}) for i in range(batch['input_ids'].size(1))]
    return [torch.sum(torch.stack(f), dim=0) for f in zip(*fishers)]

def compute_fisher_for_model(model, dataloader, expectation_wrt_logits=True):
    variables = [p for p in model.parameters() if p.requires_grad]
    fishers = [torch.zeros_like(p) for p in variables]

    n_examples = 0

    for batch in dataloader:
        n_examples += _batch_size(batch)
        batch_fishers = _compute_exact_fisher_for_batch(batch, model, variables, expectation_wrt_logits=expectation_wrt_logits)
        for f, bf in zip(fishers, batch_fishers):
            f += bf

    for i in range(len(fishers)):
        fishers[i] /= float(n_examples)

    return fishers


if __name__ == "__main__":

    seed = 0
    data_dir = "/data/yekeming/project/neurips2024/uniting4nlp/data/from_local/rte"
    task = "rte"
    model_str = "/data/yekeming/project/neurips2024/uniting4nlp/ckpt_paths/rte/rte-roberta-base-finetune-32-1e-3-seed0/finetune"
    n_examples = 128
    batch_size = 2
    fisher_path = ""

    set_seed(seed)

    # Expand the model just in case it is a path rather than
    # the name of a model from HuggingFace's repository.
    model = AutoModelForSequenceClassification.from_pretrained(
        model_str
    )
    tokenizer = AutoTokenizer.from_pretrained(model_str)
    logging.info("Model loaded")

    actual_task = "mnli" if task == "mnli-mm" else task
    dataset = load_from_disk(data_dir)

    metric = load_metric('glue', actual_task)

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

    ds = dataset.map(preprocess_function, batched=True)

    selected_indices = list(range(n_examples))
    dataloader = DataLoader(ds["train"], batch_size=batch_size, shuffle=False)

    for batch in dataloader:
        pass

    logging.info("Dataset loaded")

    logging.info("Starting Fisher computation")
    fisher_diag = compute_fisher_for_model(model, dataloader)

    logging.info("Fisher computed. Saving to file...")
    fisher_path = os.path.expanduser(fisher_path)
    save_variables_to_hdf5(fisher_diag, fisher_path)
    logging.info("Fisher saved to file")

