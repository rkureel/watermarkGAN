import torch.nn as nn
from config.watermarkganconfig import WatermarkGANConfiguration

class Critic(nn.Module):
    def __init__(self, config: WatermarkGANConfiguration):
        super(Critic, self).__init__()

        # layers = [ConvBNRelu(3, config.discriminator_channels)]
        # for _ in range(config.discriminator_blocks-1):
        #     layers.append(ConvBNRelu(config.discriminator_channels, config.discriminator_channels))

        # layers.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        # self.before_linear = nn.Sequential(*layers)
        # self.linear = nn.Linear(config.discriminator_channels, 1)

    def forward(self, image):
        X = self.before_linear(image)
        # the output is of shape b x c x 1 x 1, and we want to squeeze out the last two dummy dimensions and make
        # the tensor of shape b x c. If we just call squeeze_() it will also squeeze the batch dimension when b=1.
        X.squeeze_(3).squeeze_(2)
        X = self.linear(X)
        # X = torch.sigmoid(X)
        return X