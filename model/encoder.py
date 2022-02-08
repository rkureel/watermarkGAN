import torch
import torch.nn as nn
from .conv_relu_bn import ConvReluBN
from config.watermarkganconfig import WatermarkGANConfiguration


class Encoder(nn.Module):
    def __init__(self, config: WatermarkGANConfiguration):
        super(Encoder, self).__init__()
        self.H = config.H
        self.W = config.W
        self.conv_channels = config.encoder_channels
        self.num_blocks = config.encoder_blocks
        self.models = self._build_models()

    def _build_models(self):
        self.features = nn.Sequential(
            ConvReluBN(3, self.conv_channels)
        )

        self.conv1 = nn.Sequential(
            ConvReluBN(self.conv_channels + config.message_length, self.conv_channels),
        )

        self.conv2 = nn.Sequential(
            ConvReluBN(self.conv_channels*2 + config.message_length, self.conv_channels),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=self.conv_channels*3 + self.data_depth, out_channels=3, kernel_size=3, padding=1)
        )
        return self.features, self.conv1, self.conv2, self.conv3

    def forward(self, image, message):

        # First, add two dummy dimensions in the end of the message.
        # This is required for the .expand to work correctly
        expanded_message = message.unsqueeze(-1)
        expanded_message.unsqueeze_(-1)

        expanded_message = expanded_message.expand(-1,-1, self.H, self.W)
        encoded_image = self.models[0](image)
        
        x = encoded_image
        for layer in self.models[1:]:
            concat = torch.cat([expanded_message, x], dim=1)
            x = layer(concat)

        x = torch.cat([image, x], dim=1)
        return x
