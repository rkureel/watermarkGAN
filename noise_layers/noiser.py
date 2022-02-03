import numpy as np
import torch.nn as nn
from noise_layers.identity import Identity


class Noiser(nn.Module):
    def __init__(self, noise_layers: list, device):
        super(Noiser, self).__init__()
        self.noise_layers = [Identity()]
        # for layer in noise_layers:
        #     if type(layer) is str:
        #         if layer == 'JpegPlaceholder':
        #             self.noise_layers.append(JpegCompression(device))
        #         elif layer == 'QuantizationPlaceholder':
        #             self.noise_layers.append(Quantization(device))
        #         else:
        #             raise ValueError(f'Wrong layer placeholder string in Noiser.__init__().'
        #                              f' Expected "JpegPlaceholder" or "QuantizationPlaceholder" but got {layer} instead')
        #     else:
        #         self.noise_layers.append(layer)
        # self.noise_layers = nn.Sequential(*noise_layers)

    def forward(self, encoded_and_cover):
        random_noise_layer = np.random.choice(self.noise_layers, 1)[0]
        return random_noise_layer(encoded_and_cover)