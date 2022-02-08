import torch.nn as nn
from config.watermarkganconfig import WatermarkGANConfiguration
from noise_layers.noiser import Noiser
from .encoder import Encoder
from .decoder import Decoder


class EncoderDecoder(nn.Module):
    def __init__(self, config: WatermarkGANConfiguration, noiser: Noiser):

        super(EncoderDecoder, self).__init__()
        self.encoder = Encoder(config)
        self.noiser = noiser
        self.decoder = Decoder(config)

    def forward(self, image, message):
        encoded_image = self.encoder(image, message)
        noised_and_cover = self.noiser([encoded_image, image])
        noised_image = noised_and_cover[0]
        decoded_message = self.decoder(noised_image)
        return encoded_image, noised_image, decoded_message
