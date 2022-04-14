import argparse
import os

from steganogan.loader import DataLoader
from steganogan.models import SteganoGAN
from steganogan.critics import BasicCritic
from steganogan.decoders import DenseDecoder
from steganogan.encoders import DenseEncoder


if __name__ == "__main__":
    parent_parser = argparse.ArgumentParser(description="Parent parser")
    subparsers = parent_parser.add_subparsers(dest='command', required=True, help='Sub-parser for commands')
    
    train_parser = subparsers.add_parser('train', help='training a model')
    train_parser.add_argument('--data-dir', '-d', required=True, type=str, help='The directory where the data is stored.')
    train_parser.add_argument('--epochs', '-e', required=True, type=int, help='Number of epochs')
    train_parser.add_argument('--noise', '-n', required=True, type=str, help='Noise layers used for training. Mapping is as follows: 1-Identity, 2-Dropout, 3-Crop, 4-Cropout, 5-Gaussian Blur, 6-JPEG Compression')

    evaluate_parser = subparsers.add_parser('evaluate', help='evaluate a model')
    evaluate_parser.add_argument('--data-dir', '-d', required=True, type=str, help='The directory where data is stored')
    evaluate_parser.add_argument('--noise', '-n', required=True, type=str, help='Noise layers used for evaluation. Mapping is as follows: 1-Identity, 2-Dropout, 3-Crop, 4-Cropout, 5-Gaussian Blur, 6-JPEG Compression')
    evaluate_parser.add_argument('--path', '-p', required=True, type=str, help='The path of pretrained model.')

    args = parent_parser.parse_args()

    print(args)

    if args.command == "train":
        train_data_dir = os.path.join(args.data_dir, "train")
        val_data_dir = os.path.join(args.data_dir, "val")
        train = DataLoader(train_data_dir)
        validation = DataLoader(val_data_dir)

        steganogan = SteganoGAN(1, DenseEncoder, DenseDecoder, BasicCritic, args.noise, hidden_size=32, cuda=True, verbose=True)
        steganogan.fit(train, validation, epochs=args.epochs)
    elif args.command == "evaluate":
        val_data_dir = os.path.join(args.data_dir, "val")
        validation = DataLoader(val_data_dir)

        steganogan = SteganoGAN(1, DenseEncoder, DenseDecoder, BasicCritic, args.noise, hidden_size=32, cuda=True, verbose=True)
        steganogan.load(path=args.path)
        steganogan.evaluate(validation)
