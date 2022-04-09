from steganogan.loader import DataLoader
from steganogan.models import SteganoGAN
from steganogan.critics import BasicCritic
from steganogan.decoders import DenseDecoder
from steganogan.encoders import DenseEncoder

train = DataLoader('data/div2k/train/')
validation = DataLoader('data/div2k/val/')

steganogan = SteganoGAN(1, DenseEncoder, DenseDecoder, BasicCritic, hidden_size=32, cuda=True, verbose=True)
steganogan.fit(train, validation, epochs=200)