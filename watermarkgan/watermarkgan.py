from mimetypes import init
import torch
import gc
import inspect
import json
import os
import torch
import numpy as np
from collections import Counter
from torch.nn.functional import binary_cross_entropy_with_logits, mse_loss
from torch.optim import Adam
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

METRIC_FIELDS = [
    'val.encoder_mse',
    'val.decoder_loss',
    'val.decoder_acc',
    'val.cover_score',
    'val.generated_score',
    'val.psnr',
    'val.bpp',
    'train.encoder_mse',
    'train.decoder_loss',
    'train.decoder_acc',
    'train.cover_score',
    'train.generated_score',
]

path =os.makedirs("logs", exist_ok=True)
writer = SummaryWriter(path)

class WatermarkGAN(object):

    def _get_instance(self, class_or_instance, kwargs):
        if not inspect.isclass(class_or_instance):
            return class_or_instance
        
        argspec = inspect.getfullargspec(class_or_instance.__init__).args
        argspec.remove("self")
        init_args = {arg: kwargs[arg] for arg in argspec}
        return class_or_instance(**init_args)

    def set_device(self, cuda=True):
        if cuda and torch.cuda.is_available():
            self.cuda = True
            self.device = torch.device("cuda")
        else:
            self.cuda = False
            self.device = torch.device("cpu")
        
        if self.verbose:
            if not cuda:
                print("Using CPU device")
            elif not self.cuda:
                print("CUDA is not available. Defaulting to CPU device")
            else:
                print("Using CUDA device")
            
        self.encoder.to(self.device)
        self.decoder.to(self.device)
        self.critic.to(self.device)
    
    def __init__(self, message_length, encoder, decoder, critic, cuda=False, verbose=True, log_dir="logs", **kwargs):
        self.verbose = verbose
        self.message_length = message_length
        kwargs["message_length"] = message_length
        self.encoder = self._get_instance(encoder, kwargs)
        self.decoder = self._get_instance(decoder, kwargs)
        self.critic = self._get_instance(critic, kwargs)
        self.set_device(cuda)

        self.critic_optimizer = None
        self.decoder_optimizer = None

        self.fit_metrics = None
        self.history = list()

        self.log_dir = log_dir
        if log_dir:
            os.makedirs(self.log_dir, exist_ok=True)
            self.samples_path = os.path.join(self.log_dir, "samples")
            os.makedirs(self.samples_path, exist_ok=True)
            self.tensorlogs_path = os.path.join(self.log_dir, "watermarkgan")
            self.models_dir = os.path.join(self.log_dir, "models")
            os.makedirs(self.models_dir, exist_ok=True)

    def _random_data(self, cover):
        message = torch.Tensor(np.random.choice([0, 1], (cover.shape[0], 30))).to(self.device)
        return message
    
    def _encode_decode(self, cover, quantize=False):
        payload = self._random_data(cover)
        generated = self.encoder(cover, payload)
        if quantize:
            generated = (255.0 * (generated + 1.0)/2.0).long()
            generated = 2.0*generated.float() / 255.0 - 1.0

        decoded = self.decoder(generated)

        return generated, payload, decoded
    
    def _critic(self, image):
        return torch.mean(self.critic(image))

    def _get_optimizers(self):
        _dec_list = list(self.decoder.parameters()) + list(self.encoder.parameters())
        critic_optimizer = Adam(self.critic.parameters(), lr=1e-4)
        decoder_optimizer = Adam(_dec_list, lr=1e-4)

        return critic_optimizer, decoder_optimizer
    
    def _fit_critic(self, train, metrics):
        for cover, _ in tqdm(train, disable=not self.verbose):
            gc.collect()
            cover = cover.to(self.device)
            payload = self._random_data(cover)
            generated = self.encoder(cover, payload)
            cover_score = self._critic(cover)
            generated_score = self._critic(generated)

            self.critic_optimizer.zero_grad()
            (cover_score - generated_score).backward(retain_graph=False)
            self.critic_optimizer.step()

            for p in self.critic.parameters():
                p.data.clamp(-0.1, 0.1)
            
            metrics["train.cover_score"].append(cover_score.item())
            metrics["train.generated_score"].append(generated_score.item())
    
    def _coding_scores(self, cover, generated, payload, decoded):
        encoder_mse = mse_loss(generated, cover)
        payload = payload.unsqueeze(-1)
        payload.unsqueeze_(-1)
        payload = payload.expand(-1,-1, 128, 128)
        decoder_loss = binary_cross_entropy_with_logits(decoded, payload)
        decoder_acc = (decoded >= 0.0).eq(payload >= 0.5).sum().float() / payload.numel()
        return encoder_mse, decoder_loss, decoder_acc

    def _fit_coders(self, train, metrics):
        for cover, _ in tqdm(train, disable=not self.verbose):
            gc.collect()
            cover = cover.to(self.device)
            generated, payload, decoded = self._encode_decode(cover)
            encoder_mse, decoder_loss, decoder_acc = self._coding_scores(
                cover, generated, payload, decoded
            )
            generated_score = self._critic(generated)

            self.decoder_optimizer.zero_grad()
            (100.0*encoder_mse+decoder_loss+generated_score).backward()
            self.decoder_optimizer.step()

            metrics["train.encoder_mse"].append(encoder_mse.item())
            metrics["train.decoder_loss"].append(decoder_loss.item())
            metrics["train.decoder_acc"].append(decoder_acc.item())

    def _validate(self, validate, metrics):
        for cover, _ in tqdm(validate, disable=not self.verbose):
            gc.collect()
            cover = cover.to(self.device)
            generated, payload, decoded = self._encode_decode(cover, quantize=True)
            encoder_mse, decoder_loss, decoder_acc = self._coding_scores(
                cover, generated, payload, decoded)
            generated_score = self._critic(generated)
            cover_score = self._critic(cover)

            metrics['val.encoder_mse'].append(encoder_mse.item())
            metrics['val.decoder_loss'].append(decoder_loss.item())
            metrics['val.decoder_acc'].append(decoder_acc.item())
            metrics['val.cover_score'].append(cover_score.item())
            metrics['val.generated_score'].append(generated_score.item())
            metrics['val.psnr'].append(10 * torch.log10(4 / encoder_mse).item())
            metrics['val.bpp'].append(0)

    def fit(self, train, validate, epochs=5):
        self.save('logs/test.wmgan')
        if self.critic_optimizer is None:
            self.critic_optimizer, self.decoder_optimizer = self._get_optimizers()
            self.epochs = 0

        if self.log_dir:
            sample_cover = next(iter(validate))[0]

        total = self.epochs + epochs
        for epoch in range(1, epochs + 1):
            self.epochs += 1

            metrics = {field: list() for field in METRIC_FIELDS}

            if self.verbose:
                print('Epoch {}/{}'.format(self.epochs, total))

            self._fit_critic(train, metrics)
            self._fit_coders(train, metrics)
            self._validate(validate, metrics)

            self.fit_metrics = {k: sum(v) / len(v) for k, v in metrics.items()}
            self.fit_metrics['epoch'] = epoch

            if self.log_dir:
                self.history.append(self.fit_metrics)

                metrics_path = os.path.join(self.log_dir, 'metrics.log')
                with open(metrics_path, 'w') as metrics_file:
                    json.dump(self.history, metrics_file, indent=4)

                save_name = '{}.bpp-{:03f}.p'.format(
                    self.epochs, self.fit_metrics['val.bpp'])

                self.save(os.path.join(self.log_dir, save_name))
            
            save_path = os.path.join(self.models_dir, str(epoch) + ".wmgan")
            self.save(save_path)
            for key, val in self.fit_metrics.items():
                writer.add_scalar(key, val, epoch-1)

            if self.cuda:
                torch.cuda.empty_cache()

            gc.collect()

    def save(self, path):
        torch.save(self, path)