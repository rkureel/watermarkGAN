import argparse
import pickle
import sys
import torch
import os
import utils
import logging
from pprint import pprint
from train import train
from model.watermarkgan import WatermarkGAN
from noise_layers.noiser import Noiser
from config.watermarkganconfig import WatermarkGANConfiguration
from noise_argparser import NoiseArgParser
from config.training_options import TrainingOptions


def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    parent_parser = argparse.ArgumentParser(description="Training of WatermarkGAN nets")
    subparsers = parent_parser.add_subparsers(dest="command", help="Sub-parser for commands")
    new_run_parser = subparsers.add_parser("new", help="starts a new run")
    new_run_parser.add_argument("--data-dir", "-d", required=True, type=str, help="The directory where the data is store")
    new_run_parser.add_argument("--batch-size", "-b", required=True, type=int, help="The batch size")
    new_run_parser.add_argument("--epochs", "-e", required=True, type=int, help="Number of epochs to run the simulation")
    new_run_parser.add_argument("--name", required=True, type=str, help="Name of the experiment")
    new_run_parser.add_argument("--size", "-s", default=128, type=int, help="The size of the images (images are square so this is the height and width)")
    new_run_parser.add_argument("--message", "-m", default=30, type=int, help="The length in bits of the watermark")
    new_run_parser.add_argument("--continue-from-folder", "-c", default="", type=str, help="The folder from where to continue a previous run. Leave blank if you are starting  a new experiment")
    new_run_parser.add_argument("--noise", nargs="*", action=NoiseArgParser, help="Noise layers configuration. Use quotes when specifying configuration, e.g. 'cropout((0.55, 0.6), (0.55, 0.6))")
    new_run_parser.add_argument("--tensorboard", action="store_true", help="Use to switch on tensorboard logging")
    
    new_run_parser.set_defaults(tensorboard=False)

    continue_parser = subparsers.add_parser("continue", help="Continue a previous run")
    continue_parser.add_argument("--folder", "-f", required=True, type=str, help="Continue from the last checkpoint in this folder")
    continue_parser.add_argument("--data-dir", "-d", required=False, type=str, help="The directory where the data is stored. Specify a value only if you want to override the previous value")
    continue_parser.add_argument("--epochs", "-e", required=False, type=int, help="Number of epochs to run the simluation. Specify a value only if you want to override the previous value")

    args = parent_parser.parse_args()
    checkpoint = None
    loaded_checkpoint_filename = None

    if args.command == "continue":
        this_run_folder = args.folder
        options_file = os.path.join(this_run_folder, "options-and-config.pickle")
        train_options, hidden_config, noise_config = utils.load_options(options_file)
        checkpoint, loaded_checkpoint_filename = utils.load_last_checkpoint(os.path.join(this_run_folder, "checkpoints"))
        train_options.start_epoch = checkpoint["epoch"] + 1
        
        if args.data_dir is not None:
            train_options.train_folder = os.path.join(args.data_dir, "train")
            train_options.validation_folder = os.path.join(args.data_dir, "val")
        if args.epochs is not None:
            if train_options.start_epoch < args.epochs:
                train_options.number_of_epochs = args.epochs
            else:
                print(f"Command-line specifies of number of epochs = {args.epochs}, but folder={args.folder} already contains checkpoint for epoch = {train_options.start_epoch}")
                exit(1)
    else:
        assert args.command == "new"
        start_epoch = 1
        train_options = TrainingOptions(
            batch_size=args.batch_size,
            number_of_epochs=args.epochs,
            train_folder=os.path.join(args.data_dir, "train"),
            validation_folder=os.path.join(args.data_dir, "val"),
            runs_folder=os.path.join(".", "runs"),
            start_epoch=start_epoch,
            experiment_name=args.name
        )

        noise_config = args.noise if args.noise is not None else []
        watermarkconfig = WatermarkGANConfiguration(
            H=args.size, W=args.size,
            message_length=args.message,
            encoder_channels=32,
            decoder_channels=32,
            critic_channels=32,
        )

        this_run_folder = utils.create_folder_for_run(train_options.runs_folder, args.name)
        with open(os.path.join(this_run_folder, "options-and-config.pickle"), "wb+") as f:
            pickle.dump(train_options, f)
            pickle.dump(noise_config, f)
            pickle.dump(watermarkconfig, f)
        
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[
            logging.FileHandler(os.path.join(this_run_folder, f"{train_options.experiment_name}.log")),
            logging.StreamHandler(sys.stdout)
        ]
    )

    if (args.command == "new" and args.tensorboard) or (args.command == "continue" and os.path.isdir(os.path.join(this_run_folder, "tb-logs"))):
        logging.info("Tensorboard is enabled. Creating logger")
        from tensorboard_logger import TensorBoardLogger
        tb_logger = TensorBoardLogger(os.path.join(this_run_folder, "tb-logs"))
    else:
        tb_logger = None

    noiser = Noiser(noise_config, device)
    model = WatermarkGAN(watermarkconfig, device, noiser, tb_logger)
    
    if args.command == "continue":
        assert checkpoint is not None
        logging.info(f"Loading checkpoint from file {loaded_checkpoint_filename}")
        utils.model_from_checkpoint(model, checkpoint)

    logging.info(f"WatermarkGAN model: {model.to_string}")
    logging.info("Model Configuration:\n")
    logging.info(pprint.pformat(vars(watermarkconfig)))
    logging.info("\nNoise configuration:\n")
    logging.info(pprint.pformat(str(noise_config)))
    logging.info("\nTraining train_options:\n")
    logging.info(pprint.pformat(vars(train_options)))

    train(model, device, hidden_config, train_options, this_run_folder, tb_logger)

if __name__ == "__main__":
    main()