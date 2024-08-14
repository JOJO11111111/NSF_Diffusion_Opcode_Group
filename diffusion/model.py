import gc
import os
import math
import random
import pandas as pd
from tqdm.notebook import tqdm_notebook
import torch
import torch.nn as nn
from torch.cuda import amp
from torchmetrics import MeanMetric

from utils import *
from data import get_dataset, get_dataloader, inverse_transform

#################################
# UNET
#################################

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, total_time_steps=1000, time_emb_dims=128, time_emb_dims_exp=512):
        super().__init__()

        half_dim = time_emb_dims // 2

        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)

        ts = torch.arange(total_time_steps, dtype=torch.float32)

        emb = torch.unsqueeze(ts, dim=-1) * torch.unsqueeze(emb, dim=0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)

        self.time_blocks = nn.Sequential(
            nn.Embedding.from_pretrained(emb),
            nn.Linear(in_features=time_emb_dims, out_features=time_emb_dims_exp),
            nn.SiLU(),
            nn.Linear(in_features=time_emb_dims_exp, out_features=time_emb_dims_exp),
        )

    def forward(self, time):
        return self.time_blocks(time)

class AttentionBlock(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.channels = channels

        self.group_norm = nn.GroupNorm(num_groups=8, num_channels=channels)
        self.mhsa = nn.MultiheadAttention(embed_dim=self.channels, num_heads=4, batch_first=True)

    def forward(self, x):
        # B, _, H, W = x.shape
        # h = self.group_norm(x)
        # h = h.reshape(B, self.channels, H * W).swapaxes(1, 2)  # [B, C, H, W] --> [B, C, H * W] --> [B, H*W, C]
        # h, _ = self.mhsa(h, h, h)  # [B, H*W, C]
        # h = h.swapaxes(2, 1).view(B, self.channels, H, W)  # [B, C, H*W] --> [B, C, H, W]

        # B, _, L = x.shape #B: batch size, C: channels, L: length
        h = self.group_norm(x)
        h = h.swapaxes(1, 2)  # [B, L, C]
        h, _ = self.mhsa(h, h, h)
        h = h.swapaxes(2, 1)  # [B, C, L]
        return x + h

class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels, dropout_rate=0.1, time_emb_dims=512, apply_attention=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.act_fn = nn.SiLU()
        # Group 1
        self.normlize_1 = nn.GroupNorm(num_groups=8, num_channels=self.in_channels)
        # self.conv_1 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=3, stride=1, padding="same")
        self.conv_1 = nn.Conv1d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=3, stride=1, padding="same")

        # Group 2 time embedding
        self.dense_1 = nn.Linear(in_features=time_emb_dims, out_features=self.out_channels)

        # Group 3
        self.normlize_2 = nn.GroupNorm(num_groups=8, num_channels=self.out_channels)
        # self.dropout = nn.Dropout2d(p=dropout_rate)
        # self.conv_2 = nn.Conv2d(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=3, stride=1, padding="same")
        self.dropout = nn.Dropout1d(p=dropout_rate)
        self.conv_2 = nn.Conv1d(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=3, stride=1, padding="same")

        if self.in_channels != self.out_channels:
            # self.match_input = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=1, stride=1)
            self.match_input = nn.Conv1d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=1, stride=1)
        else:
            self.match_input = nn.Identity()

        if apply_attention:
            self.attention = AttentionBlock(channels=self.out_channels)
        else:
            self.attention = nn.Identity()

    def forward(self, x, t):
        # group 1
        h = self.act_fn(self.normlize_1(x))
        h = self.conv_1(h)

        # group 2
        # add in timestep embedding
        h += self.dense_1(self.act_fn(t))[:, :, None] #[:, :, None, None]

        # group 3
        h = self.act_fn(self.normlize_2(h))
        h = self.dropout(h)
        h = self.conv_2(h)

        # Residual and attention
        h = h + self.match_input(x)
        h = self.attention(h)

        return h

class DownSample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # self.downsample = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=2, padding=1)
        self.downsample = nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x, *args):
        return self.downsample(x)

class UpSample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            # nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x, *args):
        return self.upsample(x)

class UNet(nn.Module):
    def __init__(
        self,
        input_channels=1,
        output_channels=1,
        num_res_blocks=2,
        base_channels=128,
        base_channels_multiples=(1, 2, 4, 8),
        apply_attention=(False, False, True, False),
        dropout_rate=0.1,  ## 0.01
        time_multiple=4, ##2, 8
    ):
        super().__init__()

        time_emb_dims_exp = base_channels * time_multiple
        self.time_embeddings = SinusoidalPositionEmbeddings(time_emb_dims=base_channels, time_emb_dims_exp=time_emb_dims_exp)

        # self.first = nn.Conv2d(in_channels=input_channels, out_channels=base_channels, kernel_size=3, stride=1, padding="same")
        ## N: Batch size, C: Channels, L: length
        ## Input to Conv1d: (N, C_in, L)   -> should be [128, 1, 256]
        ## Output to Conv1d: (N, C_out, L) -> should be [128, 128, 256]
        self.first = nn.Conv1d(in_channels=input_channels, out_channels=base_channels, kernel_size=3, stride=1, padding="same")

        num_resolutions = len(base_channels_multiples)

        # Encoder part of the UNet. Dimension reduction.
        self.encoder_blocks = nn.ModuleList()
        curr_channels = [base_channels]
        in_channels = base_channels

        for level in range(num_resolutions):
            out_channels = base_channels * base_channels_multiples[level]

            for _ in range(num_res_blocks):

                block = ResnetBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    dropout_rate=dropout_rate,
                    time_emb_dims=time_emb_dims_exp,
                    apply_attention=apply_attention[level],
                )
                self.encoder_blocks.append(block)

                in_channels = out_channels
                curr_channels.append(in_channels)

            if level != (num_resolutions - 1):
                self.encoder_blocks.append(DownSample(channels=in_channels))
                curr_channels.append(in_channels)

        # Bottleneck in between
        self.bottleneck_blocks = nn.ModuleList(
            (
                ResnetBlock(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    dropout_rate=dropout_rate,
                    time_emb_dims=time_emb_dims_exp,
                    apply_attention=True,
                ),
                ResnetBlock(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    dropout_rate=dropout_rate,
                    time_emb_dims=time_emb_dims_exp,
                    apply_attention=False,
                ),
            )
        )

        # Decoder part of the UNet. Dimension restoration with skip-connections.
        self.decoder_blocks = nn.ModuleList()

        for level in reversed(range(num_resolutions)):
            out_channels = base_channels * base_channels_multiples[level]

            for _ in range(num_res_blocks + 1):
                encoder_in_channels = curr_channels.pop()
                block = ResnetBlock(
                    in_channels=encoder_in_channels + in_channels,
                    out_channels=out_channels,
                    dropout_rate=dropout_rate,
                    time_emb_dims=time_emb_dims_exp,
                    apply_attention=apply_attention[level],
                )

                in_channels = out_channels
                self.decoder_blocks.append(block)

            if level != 0:
                self.decoder_blocks.append(UpSample(in_channels))

        self.final = nn.Sequential(
            nn.GroupNorm(num_groups=8, num_channels=in_channels),
            nn.SiLU(),
            # nn.Conv2d(in_channels=in_channels, out_channels=output_channels, kernel_size=3, stride=1, padding="same"),
            nn.Conv1d(in_channels=in_channels, out_channels=output_channels, kernel_size=3, stride=1, padding="same"),
        )

    def forward(self, x, t):

        time_emb = self.time_embeddings(t)
        # print(f'Time Emb Shape: {time_emb.shape}')
        # print(f'X Shape: {x.shape}')

        h = self.first(x)
        outs = [h]

        for layer in self.encoder_blocks:
            h = layer(h, time_emb)
            outs.append(h)

        for layer in self.bottleneck_blocks:
            h = layer(h, time_emb)

        for layer in self.decoder_blocks:
            if isinstance(layer, ResnetBlock):
                out = outs.pop()
                h = torch.cat([h, out], dim=1)
            h = layer(h, time_emb)

        h = self.final(h)

        return h

#################################
# Diffusion
#################################

class SimpleDiffusion:
    def __init__(
        self,
        num_diffusion_timesteps=1000, 
        img_shape=(1, 256),
        device="cpu",
        time_schedule = 0,
    ):
        self.num_diffusion_timesteps = num_diffusion_timesteps
        self.img_shape = img_shape
        self.device = device
        self.get_betas = time_schedule

        self.initialize()

    def initialize(self):
        # BETAs & ALPHAs required at different places in the Algorithm.
        if (self.get_betas == 0):
          self.beta  = self.get_betas_linear()
        elif (self.get_betas == 1):
          self.beta = self.get_betas_cosine()
        elif (self.get_betas == 2):
          self.beta = self.get_betas_cosine_extended()
          
        self.alpha = 1 - self.beta

        self_sqrt_beta                       = torch.sqrt(self.beta)
        self.alpha_cumulative                = torch.cumprod(self.alpha, dim=0)
        self.sqrt_alpha_cumulative           = torch.sqrt(self.alpha_cumulative)
        self.one_by_sqrt_alpha               = 1. / torch.sqrt(self.alpha)
        self.sqrt_one_minus_alpha_cumulative = torch.sqrt(1 - self.alpha_cumulative)

    def get_betas_linear(self):
        """linear schedule, proposed in original ddpm paper"""
        scale = 1000 / self.num_diffusion_timesteps
        beta_start = scale * 1e-4
        beta_end = scale * 0.02
        return torch.linspace(
            beta_start,
            beta_end,
            self.num_diffusion_timesteps,
            dtype=torch.float32,
            device=self.device,
        )
    def get_betas_cosine(self):
        """
        cosine schedule as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
        """
        timesteps = self.num_diffusion_timesteps
        s = 0.008

        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.999).to(torch.float32).to(self.device)
    def get_betas_cosine_extended(self):
        """
        cosine schedule not using last 100ts
        """
        timesteps = self.num_diffusion_timesteps
        s = 0.008

        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
        timesteps += 100
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.999).to(torch.float32).to(self.device)

def forward_diffusion(sd: SimpleDiffusion, x0: torch.Tensor, timesteps: torch.Tensor):
    # print('---------------------------------------')
    # print(f'IN FORWARD DIFFUSION')
    # print('---------------------------------------')
    eps = torch.randn_like(x0)  # Noise
    mean    = get(sd.sqrt_alpha_cumulative, t=timesteps) * x0  # Image scaled
    std_dev = get(sd.sqrt_one_minus_alpha_cumulative, t=timesteps) # Noise scaled
    sample  = mean + std_dev * eps

    return sample, eps  # return ... , gt noise --> model predicts this)

# Algorithm 1: Training

def train_one_epoch(model, sd, loader, optimizer, scaler, loss_fn, epoch, 
                   base_config, training_config):

    loss_record = MeanMetric()
    model.train()

    with tqdm_notebook(total=len(loader), dynamic_ncols=True) as tq:
        tq.set_description(f"Train :: Epoch: {epoch}/{training_config.NUM_EPOCHS}")

        for x0s in loader:
            tq.update(1)

            ## change to timesteps/2 to max out at half images
            ts = torch.randint(low=1, high=int(training_config.TIMESTEPS / 2), size=(x0s.shape[0],), device=base_config.DEVICE)
            xts, gt_noise = forward_diffusion(sd, x0s, ts)

            with amp.autocast():
                pred_noise = model(xts, ts)
                loss = loss_fn(gt_noise, pred_noise)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()

            # scaler.unscale_(optimizer)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()

            loss_value = loss.detach().item()
            loss_record.update(loss_value)

            tq.set_postfix_str(s=f"Loss: {loss_value:.4f}")

        mean_loss = loss_record.compute().item()

        tq.set_postfix_str(s=f"Epoch Loss: {mean_loss:.4f}")

    return mean_loss

# Algorithm 2: Sampling

@torch.inference_mode()
def reverse_diffusion(model, sd, dataset, timesteps=1000, img_shape=(1, 256),
                      num_images=5, nrow=8, device="cpu", **kwargs):

    # get images
    indices = random.sample(range(len(dataset)), num_images)
    x0s = [dataset[i] for i in indices]
    x0s = torch.stack(x0s).to(device)
    # create tensor of timesteps (1 per image)
    ts = ( torch.ones(size=(x0s.shape[0],), device=device) * int(timesteps / 2) ).to(torch.int64)
    # create noised samples
    xts, _ = forward_diffusion(sd, x0s, ts)
    x = xts.to(device)

    model.eval()

    for time_step in tqdm_notebook(iterable=reversed(range(1, timesteps)),
                          total=timesteps-1, dynamic_ncols=False,
                          desc="Sampling :: ", position=0):

        ts = torch.ones(num_images, dtype=torch.long, device=device) * time_step
        z = torch.randn_like(x) if time_step > 1 else torch.zeros_like(x)

        predicted_noise = model(x, ts)

        beta_t                            = get(sd.beta, ts)
        one_by_sqrt_alpha_t               = get(sd.one_by_sqrt_alpha, ts)
        sqrt_one_minus_alpha_cumulative_t = get(sd.sqrt_one_minus_alpha_cumulative, ts)

        x = (
            one_by_sqrt_alpha_t
            * (x - (beta_t / sqrt_one_minus_alpha_cumulative_t) * predicted_noise)
            + torch.sqrt(beta_t) * z
        )

    # Save the image at the final timestep of the reverse process.
    x = inverse_transform(x).type(torch.float32)
    x = x.squeeze(dim=1) # turn from [B, C, L] to [B, L]
    x = pd.DataFrame(x.cpu().numpy())
    x.to_csv(kwargs['save_path'], index=False, header = False)

    return None


def train_model(BaseConfig, ModelConfig, TrainingConfig):
    model = UNet(
        input_channels          = TrainingConfig.IMG_SHAPE[0],
        output_channels         = TrainingConfig.IMG_SHAPE[0],
        base_channels           = ModelConfig.BASE_CH,
        base_channels_multiples = ModelConfig.BASE_CH_MULT,
        apply_attention         = ModelConfig.APPLY_ATTENTION,
        dropout_rate            = ModelConfig.DROPOUT_RATE,
        time_multiple           = ModelConfig.TIME_EMB_MULT,
    )
    model.to(BaseConfig.DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=TrainingConfig.LR)

    dataloader = get_dataloader(
        dataset_name  = BaseConfig.DATASET,
        batch_size    = TrainingConfig.BATCH_SIZE,
        device        = BaseConfig.DEVICE,
        pin_memory    = True,
        num_workers   = TrainingConfig.NUM_WORKERS,
    )

    loss_fn = nn.MSELoss()

    sd = SimpleDiffusion(
        num_diffusion_timesteps = TrainingConfig.TIMESTEPS,
        img_shape               = TrainingConfig.IMG_SHAPE,
        device                  = BaseConfig.DEVICE,
    )

    scaler = amp.GradScaler()

    total_epochs = TrainingConfig.NUM_EPOCHS + 1
    log_dir, checkpoint_dir, version = setup_log_directory(config=BaseConfig)

    file_name = os.path.join("Logs_Checkpoints", "params")
    os.makedirs(file_name, exist_ok=True)
    file_name = os.path.join(file_name, str(f'{version}'))
    write_config_to_file(file_name, BaseConfig, ModelConfig, TrainingConfig)

    os.makedirs("Logs_Checkpoints/loss", exist_ok=True)
    loss_file = os.path.join("Logs_Checkpoints/loss", str(f'{version}.csv'))

    ext = ".csv"

    for epoch in tqdm_notebook( range(1, total_epochs), desc="Epoch"):
        torch.cuda.empty_cache()
        gc.collect()

        # Algorithm 1: Training
        epoch_loss = train_one_epoch(model, sd, dataloader, optimizer, scaler, loss_fn, epoch=epoch,
                        base_config=BaseConfig, training_config=TrainingConfig)
        save_epoch_loss_to_csv(epoch, epoch_loss, loss_file)

        if epoch % 10 == 0:
            save_path = os.path.join(log_dir, f"{epoch}{ext}")

            # Algorithm 2: Sampling
            reverse_diffusion(model, sd, dataset=get_dataset(dataset_name=BaseConfig.DATASET), timesteps=TrainingConfig.TIMESTEPS, num_images=32,
                save_path=save_path, img_shape=TrainingConfig.IMG_SHAPE, device=BaseConfig.DEVICE,
            )

            # clear_output()
            checkpoint_dict = {
                "opt": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "model": model.state_dict()
            }
            torch.save(checkpoint_dict, os.path.join(checkpoint_dir, "ckpt.tar"))
            del checkpoint_dict

    
    save_path = os.path.join(log_dir, "final.csv")
    reverse_diffusion(
        model,
        sd,
        dataset=get_dataset(dataset_name=BaseConfig.DATASET),
        timesteps=TrainingConfig.TIMESTEPS,
        num_images=100,
        save_path=save_path,
        img_shape=TrainingConfig.IMG_SHAPE,
        device=BaseConfig.DEVICE,
        nrow=8,
    )
    return save_path

def generate_samples(BaseConfig, ModelConfig, TrainingConfig, log_folder, version, num_samples):
    # set up model
    model = UNet(
        input_channels          = TrainingConfig.IMG_SHAPE[0],
        output_channels         = TrainingConfig.IMG_SHAPE[0],
        base_channels           = ModelConfig.BASE_CH,
        base_channels_multiples = ModelConfig.BASE_CH_MULT,
        apply_attention         = ModelConfig.APPLY_ATTENTION,
        dropout_rate            = ModelConfig.DROPOUT_RATE,
        time_multiple           = ModelConfig.TIME_EMB_MULT,
    )
    checkpoint_dir = os.path.join(log_folder, "checkpoints")
    checkpoint_dir = os.path.join(checkpoint_dir, str(f"version_{version}"))
    
    model.load_state_dict(torch.load(os.path.join(checkpoint_dir, "ckpt.tar"), map_location='cpu')['model'])

    model.to(BaseConfig.DEVICE)

    sd = SimpleDiffusion(
        num_diffusion_timesteps = TrainingConfig.TIMESTEPS,
        img_shape               = TrainingConfig.IMG_SHAPE,
        device                  = BaseConfig.DEVICE,
    )

    log_dir = os.path.join(log_folder, "Inference")
    log_dir = os.path.join(log_dir, str(f"version_{version}"))
    
    os.makedirs(log_dir, exist_ok=True)

    ext = ".csv"
    filename = f"{BaseConfig.DATASET}104_{num_samples}samples{ext}"

    save_path = os.path.join(log_dir, filename)

    reverse_diffusion(
        model,
        sd,
        dataset=get_dataset(dataset_name=BaseConfig.DATASET),
        num_images=num_samples,
        save_path=save_path,
        timesteps=1000,
        img_shape=TrainingConfig.IMG_SHAPE,
        device=BaseConfig.DEVICE,
        nrow=8,
    )
    print(save_path)