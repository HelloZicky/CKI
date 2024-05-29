import os
import logging
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from absl import app, flags
from torch.utils.data import DataLoader
import h5py_util
import data
import fisher

FLAGS = flags.FLAGS
flags.DEFINE_string('model', None, 'Model path or name')
flags.DEFINE_bool('from_pt', False, 'Load model from PyTorch checkpoint')
flags.DEFINE_string('glue_task', None, 'GLUE task name')
flags.DEFINE_string('split', 'train', 'Dataset split')
flags.DEFINE_integer('sequence_length', 128, 'Maximum sequence length')
flags.DEFINE_integer('n_examples', 100, 'Number of examples')
flags.DEFINE_integer('batch_size', 8, 'Batch size')
flags.DEFINE_string('fisher_path', None, 'Path to save Fisher information')

def main(_):
    # Expand the model just in case it is a path rather than
    # the name of a model from HuggingFace's repository.
    model_str = os.path.expanduser(FLAGS.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_str
    )
    tokenizer = AutoTokenizer.from_pretrained(model_str)
    logging.info("Model loaded")

    ds = data.load_glue_dataset(
        task=FLAGS.glue_task,
        split=FLAGS.split,
        tokenizer=tokenizer,
        max_length=FLAGS.sequence_length,
    )
    ds = ds.select(range(FLAGS.n_examples))
    dataloader = DataLoader(ds, batch_size=FLAGS.batch_size, shuffle=False)
    logging.info("Dataset loaded")

    logging.info("Starting Fisher computation")
    fisher_diag = fisher.compute_fisher_for_model(model, dataloader)

    logging.info("Fisher computed. Saving to file...")
    fisher_path = os.path.expanduser(FLAGS.fisher_path)
    h5py_util.save_variables_to_hdf5(fisher_diag, fisher_path)
    logging.info("Fisher saved to file")


if __name__ == "__main__":
    app.run(main)
