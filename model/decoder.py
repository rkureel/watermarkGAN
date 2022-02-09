import torch
import torch.nn as nn
from config.watermarkganconfig import WatermarkGANConfiguration
from .conv_relu_bn import ConvReluBN

class Decoder(nn.Module):
    def __init__(self, config: WatermarkGANConfiguration):

        super(Decoder, self).__init__()
        self.channels = config.decoder_channels
        self.message_length = config.message_length
        self.models = self._build_models()

        self.linear = nn.Linear(config.message_length, config.message_length)

    def _build_models(self):
        self.conv1 = nn.Sequential(
            ConvReluBN(3, self.channels)
        )
        
        self.conv2 = nn.Sequential(
            ConvReluBN(self.channels, self.channels)
        )

        self.conv3 = nn.Sequential(
            ConvReluBN(self.channels*2, self.channels)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=self.channels*3, out_channels=self.message_length, kernel_size=3, padding=1)
        )

        return self.conv1, self.conv2, self.conv3, self.conv4

    def forward(self, image_with_wm):
        x = self.models[0](image_with_wm)
        x_list = [x]
        for layer in self.models[1:]:
            concat = torch.cat(x_list, dim=1)
            x_list.append(x)

        return x
