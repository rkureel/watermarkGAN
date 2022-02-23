import torch
import os
from time import time
from watermarkgan.loader import DataLoader
from watermarkgan.watermarkgan import WatermarkGAN
from watermarkgan.encoder import Encoder
from watermarkgan.decoder import Decoder
from watermarkgan.critic import Critic

def main():
    torch.manual_seed(42)
    timestamp = int(time())
    
    train = DataLoader(os.path.join("data", "div2k", "train"), shuffle=True)
    validation = DataLoader(os.path.join("data", "div2k", "val"), shuffle=False)
    
    encoder = Encoder
    decoder = Decoder

    watermarkgan = WatermarkGAN(
        message_length=30,
        encoder=encoder,
        decoder=decoder,
        critic=Critic,
        cuda=True,
        verbose=True,
    )

    watermarkgan.fit(train, validation, epochs=1)


if __name__ == "__main__":
    main()