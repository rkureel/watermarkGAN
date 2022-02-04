import torch
import torch.nn as nn
from .conv_relu_nb import ConvReluNB
from config.watermarkganconfig import WatermarkGANConfiguration


class Encoder(nn.Module):
    def __init__(self, config: WatermarkGANConfiguration):
        super(Encoder, self).__init__()
        self.H = config.H
        self.W = config.W
        self.conv_channels = config.encoder_channels
        self.num_blocks = config.encoder_blocks

        self.features = nn.Sequential(
            ConvReluNB(3, self.conv_channels)
        )

        self.layers = nn.Sequential(
            ConvReluNB(self.conv_channels + 3 + config.message_length, self.conv_channels),
            ConvReluNB(self.conv_channels, self.conv_channels),
            nn.Conv2d(self.conv_channels, 3, kernel_size=3),
            nn.Tanh(),
        )

    def forward(self, image, message):

        # First, add two dummy dimensions in the end of the message.
        # This is required for the .expand to work correctly
        expanded_message = message.unsqueeze(-1)
        expanded_message.unsqueeze_(-1)

        expanded_message = expanded_message.expand(-1,-1, self.H, self.W)
        encoded_image = self.features(image)
        # concatenate expanded message and image
        concat = torch.cat([expanded_message, encoded_image, image], dim=1)
        im_w = self.layers(concat)
        return im_w