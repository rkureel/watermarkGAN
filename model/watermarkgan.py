import numpy as np
import torch
import torch.nn as nn

from config.watermarkganconfig import WatermarkGANConfiguration
from noise_layers.noiser import Noiser
from .encoder_decoder import EncoderDecoder
from .critic import Critic


class WatermarkGAN:
    def __init__(self, configuration: WatermarkGANConfiguration, device: torch.device, noiser: Noiser, tb_logger):
        self.device = device
        self.encoder_decoder = EncoderDecoder(configuration, noiser).to(self.device)
        self.critic = Critic(configuration).to(self.device)
        self.optimizer_enc_dec = torch.optim.Adam(self.encoder_decoder.parameters())
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters())
        self.config = configuration
        self.bce_with_logits_loss = nn.BCEWithLogitsLoss().to(self.device)
        self.mse_loss = nn.MSELoss().to(self.device)

        # Defined the labels used for training the critic/adversarial loss
        self.cover_label = 1
        self.encoded_label = 0
        self.tb_logger = tb_logger
        if tb_logger is not None:
            from tensorboard_logger import TensorBoardLogger
            encoder_final = self.encoder_decoder.encoder._modules['final_layer']
            encoder_final.weight.register_hook(tb_logger.grad_hook_by_name('grads/encoder_out'))
            decoder_final = self.encoder_decoder.decoder._modules['linear']
            decoder_final.weight.register_hook(tb_logger.grad_hook_by_name('grads/decoder_out'))
            critic_final = self.critic._modules['linear']
            critic_final.weight.register_hook(tb_logger.grad_hook_by_name('grads/critic_out'))

    def train_on_batch(self, batch: list):
        images, messages = batch

        batch_size = images.shape[0]
        self.encoder_decoder.train()
        self.critic.train()
        with torch.enable_grad():
            # ---------------- Train the critic -----------------------------
            self.optimizer_critic.zero_grad()
            # train on cover
            d_target_label_cover = torch.full((batch_size, 1), self.cover_label, device=self.device)
            d_target_label_encoded = torch.full((batch_size, 1), self.encoded_label, device=self.device)
            g_target_label_encoded = torch.full((batch_size, 1), self.cover_label, device=self.device)

            d_on_cover = self.critic(images)
            d_loss_on_cover = self.bce_with_logits_loss(d_on_cover, d_target_label_cover)
            d_loss_on_cover.backward()

            # train on fake
            encoded_images, noised_images, decoded_messages = self.encoder_decoder(images, messages)
            d_on_encoded = self.critic(encoded_images.detach())
            d_loss_on_encoded = self.bce_with_logits_loss(d_on_encoded, d_target_label_encoded)

            d_loss_on_encoded.backward()
            self.optimizer_critic.step()

            # --------------Train the generator (encoder-decoder) ---------------------
            self.optimizer_enc_dec.zero_grad()
            # target label for encoded images should be 'cover', because we want to fool the critic
            d_on_encoded_for_enc = self.critic(encoded_images)
            g_loss_adv = self.bce_with_logits_loss(d_on_encoded_for_enc, g_target_label_encoded)

            # if self.vgg_loss == None:
            #     g_loss_enc = self.mse_loss(encoded_images, images)
            # else:
            #     vgg_on_cov = self.vgg_loss(images)
            #     vgg_on_enc = self.vgg_loss(encoded_images)
            #     g_loss_enc = self.mse_loss(vgg_on_cov, vgg_on_enc)

            g_loss_dec = self.mse_loss(decoded_messages, messages)
            # g_loss = self.config.adversarial_loss * g_loss_adv + self.config.encoder_loss * g_loss_enc \
            #          + self.config.decoder_loss * g_loss_dec

            # g_loss.backward()
            self.optimizer_enc_dec.step()

        decoded_rounded = decoded_messages.detach().cpu().numpy().round().clip(0, 1)
        bitwise_avg_err = np.sum(np.abs(decoded_rounded - messages.detach().cpu().numpy())) / (
                batch_size * messages.shape[1])

        losses = {
            # 'loss           ': g_loss.item(),
            # 'encoder_mse    ': g_loss_enc.item(),
            'dec_mse        ': g_loss_dec.item(),
            'bitwise-error  ': bitwise_avg_err,
            'adversarial_bce': g_loss_adv.item(),
            'discr_cover_bce': d_loss_on_cover.item(),
            'discr_encod_bce': d_loss_on_encoded.item()
        }
        return losses, (encoded_images, noised_images, decoded_messages)

    def validate_on_batch(self, batch: list):
        """
        Runs validation on a single batch of data consisting of images and messages
        :param batch: batch of validation data, in form [images, messages]
        :return: dictionary of error metrics from Encoder, Decoder, and critic on the current batch
        """
        # if TensorboardX logging is enabled, save some of the tensors.
        if self.tb_logger is not None:
            encoder_final = self.encoder_decoder.encoder._modules['final_layer']
            self.tb_logger.add_tensor('weights/encoder_out', encoder_final.weight)
            decoder_final = self.encoder_decoder.decoder._modules['linear']
            self.tb_logger.add_tensor('weights/decoder_out', decoder_final.weight)
            critic_final = self.critic._modules['linear']
            self.tb_logger.add_tensor('weights/critic_out', critic_final.weight)

        images, messages = batch

        batch_size = images.shape[0]

        self.encoder_decoder.eval()
        self.critic.eval()
        with torch.no_grad():
            d_target_label_cover = torch.full((batch_size, 1), self.cover_label, device=self.device)
            d_target_label_encoded = torch.full((batch_size, 1), self.encoded_label, device=self.device)
            g_target_label_encoded = torch.full((batch_size, 1), self.cover_label, device=self.device)

            d_on_cover = self.critic(images)
            d_loss_on_cover = self.bce_with_logits_loss(d_on_cover, d_target_label_cover)

            encoded_images, noised_images, decoded_messages = self.encoder_decoder(images, messages)

            d_on_encoded = self.critic(encoded_images)
            d_loss_on_encoded = self.bce_with_logits_loss(d_on_encoded, d_target_label_encoded)

            d_on_encoded_for_enc = self.critic(encoded_images)
            g_loss_adv = self.bce_with_logits_loss(d_on_encoded_for_enc, g_target_label_encoded)

            # if self.vgg_loss is None:
            #     g_loss_enc = self.mse_loss(encoded_images, images)
            # else:
            #     vgg_on_cov = self.vgg_loss(images)
            #     vgg_on_enc = self.vgg_loss(encoded_images)
            #     g_loss_enc = self.mse_loss(vgg_on_cov, vgg_on_enc)

            # g_loss_dec = self.mse_loss(decoded_messages, messages)
            # g_loss = self.config.adversarial_loss * g_loss_adv + self.config.encoder_loss * g_loss_enc \
            #          + self.config.decoder_loss * g_loss_dec

        decoded_rounded = decoded_messages.detach().cpu().numpy().round().clip(0, 1)
        bitwise_avg_err = np.sum(np.abs(decoded_rounded - messages.detach().cpu().numpy())) / (
                batch_size * messages.shape[1])

        losses = {
            # 'loss           ': g_loss.item(),
            # 'encoder_mse    ': g_loss_enc.item(),
            # 'dec_mse        ': g_loss_dec.item(),
            'bitwise-error  ': bitwise_avg_err,
            'adversarial_bce': g_loss_adv.item(),
            'discr_cover_bce': d_loss_on_cover.item(),
            'discr_encod_bce': d_loss_on_encoded.item()
        }
        return losses, (encoded_images, noised_images, decoded_messages)

    def to_string(self):
        return '{}\n{}'.format(str(self.encoder_decoder), str(self.critic))