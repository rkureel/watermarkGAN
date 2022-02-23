import torch
import torch.nn as nn
from .conv_relu_bn import ConvReluBN

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.channels = 32
        self.message_length = 30
        self.models = self._build_models()

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
            x = layer(torch.cat(x_list, dim=1))
            x_list.append(x)

        return x