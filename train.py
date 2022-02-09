import logging
import torch
import utils
import time
import numpy as np
import os
from average_meter import AverageMeter
from collections import defaultdict
from model.watermarkgan import WatermarkGAN
from config.watermarkganconfig import WatermarkGANConfiguration
from config.training_options import TrainingOptions

def train(
    model: WatermarkGAN,
    device: torch.device,
    watermark_config: WatermarkGANConfiguration,
    training_options: TrainingOptions,
    this_run_folder: str,
    tb_logger
):
    train_data, val_data = utils.get_data_loaders(watermark_config, training_options)
    file_count = len(train_data.dataset)
    if file_count % training_options.batch_size == 0:
        steps_in_epoch = file_count // training_options.batch_size
    else:
        steps_in_epoch = file_count // training_options.batch_size + 1
    
    print_each = 10
    images_to_save = 8
    saved_images_size = 512, 512

    for epoch in range(training_options.start_epoch, training_options.number_of_epochs+1):
        logging.info("\nStarting epoch {}/{}".format(epoch, training_options.number_of_epochs))
        logging.info("Batch size = {}\nSteps in epoch = {}".format(training_options.batch_size, steps_in_epoch))
        training_losses = defaultdict(AverageMeter)
        epoch_start = time.time()
        step = 1
        for image, _ in train_data:
            image = image.to(device)
            message = torch.Tensor(np.random.choice([0, 1], (image.shape[0], watermark_config.message_length))).to(device)
            print(message.shape)