class WatermarkGANConfiguration():
    def __init__(self, H: int, W: int, 
                 message_length: int,
                 encoder_channels: int,
                 decoder_channels: int,
                 critic_channels: int):
        self.H = H
        self.W = W
        self.message_length = message_length
        self.encoder_channels = encoder_channels
        self.decoder_channels = decoder_channels
        self.critic_channels = critic_channels