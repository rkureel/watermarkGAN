import torch
import os
from watermarkgan.loader import DataLoader
from watermarkgan.watermarkgan import WatermarkGAN

def main():
    train = DataLoader(os.path.join("data", "div2k", "train"), shuffle=True)
    validation = DataLoader(os.path.join("data", "div2k", "val"), shuffle=False)
    watermarkgan = WatermarkGAN(
        message_length=30,
        encoder=None,
        decoder=None,
        critic=None,
        cuda=False,
        verbose=True,
    )
    watermarkgan.fit(train, validation, epochs=1)


if __name__ == "__main__":
    main()