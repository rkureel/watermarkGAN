import torch
import torch.nn as nn
from .conv_relu_bn import ConvReluBN


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.H = 128
        self.W = 128
        self.conv_channels = 32
        self.message_length = 30
        self.models = self._build_models()

    def _build_models(self):
        self.features = nn.Sequential(
            ConvReluBN(3, self.conv_channels)
        )

        self.conv1 = nn.Sequential(
            ConvReluBN(self.conv_channels + self.message_length, self.conv_channels),
        )

        self.conv2 = nn.Sequential(
            ConvReluBN(self.conv_channels + self.message_length, self.conv_channels),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=self.conv_channels + self.message_length, out_channels=3, kernel_size=3, padding=1)
        )
        return self.features, self.conv1, self.conv2, self.conv3

    def forward(self, image, message):

        expanded_message = message.unsqueeze(-1)
        expanded_message.unsqueeze_(-1)
        expanded_message = expanded_message.expand(-1,-1, self.H, self.W)
    
        encoded_image = self.models[0](image)
        x = encoded_image
        for layer in self.models[1:]:
            concat = torch.cat([expanded_message, x], dim=1)
            x = layer(concat)
        x = image + x
        return x