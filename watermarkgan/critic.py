import torch.nn as nn
from .conv_relu_bn import ConvReluBN

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.models = self._build_models()
        self.critic_channels = 32
        # layers = [ConvBNRelu(3, config.discriminator_channels)]
        # for _ in range(config.discriminator_blocks-1):
        #     layers.append(ConvBNRelu(config.discriminator_channels, config.discriminator_channels))

        # layers.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        # self.before_linear = nn.Sequential(*layers)
        # self.linear = nn.Linear(config.discriminator_channels, 1)

    def _build_models(self):
        return nn.Sequential(
            ConvReluBN(3, self.critic_channels),
            ConvReluBN(self.critic_channels, self.critic_channels),
            ConvReluBN(self.critic_channels, self.critic_channels),
            nn.Conv2d(self.critic_channels, 1, 3)
        )

    def forward(self, image):
        # X = self.before_linear(image)
        # # the output is of shape b x c x 1 x 1, and we want to squeeze out the last two dummy dimensions and make
        # # the tensor of shape b x c. If we just call squeeze_() it will also squeeze the batch dimension when b=1.
        # X.squeeze_(3).squeeze_(2)
        # X = self.linear(X)
        # # X = torch.sigmoid(X)
        # return X
        x = self.models(image)
        return x