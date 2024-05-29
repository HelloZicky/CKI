import logging
import os

import torch


logger = logging.getLogger(__name__)


def get_device():
    try:
        logger.info(torch.cuda.get_device_name(torch.cuda.current_device()))
        device = torch.device("cuda")
    except:
        device = torch.device("cpu")

    return device
