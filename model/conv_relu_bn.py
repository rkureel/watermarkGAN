import torch.nn as nn


class ConvReluBN(nn.Module):
    def __init__(self, channels_in, channels_out, stride=1):
        super(ConvReluBN, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, 3, stride, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(channels_out)
        )
    
    def forward(self, x):
        return self.layers(x)
