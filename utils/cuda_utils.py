"""CUDA utilities"""

import torch.cuda as cuda
import torch
import logging

def get_cuda_device_with_info():
    """Prints detailes info about available CUDA devices and returns first available cuda device"""

    logging.info("### CUDA ###")
    logging.info(f"Current CUDA device id:{cuda.current_device()}")
    logging.info(f"CUDA Devices count:{cuda.device_count()}")

    # Find a better way to do this

    i = 0
    while i < cuda.device_count():
        logging.info(f"CUDA device id:{i}, | name:{cuda.get_device_name(i)}")
        i += 1
    logging.info("############")

    gpu_device = torch.device('cuda')
    logging.info(f"Using: {cuda.get_device_name(gpu_device)}\n")

    return gpu_device