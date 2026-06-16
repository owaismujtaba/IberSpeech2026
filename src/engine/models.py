import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat, einsum
import braindecode.models as bmodels

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        return output

class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, expand=2, dt_rank='auto', d_conv=4, conv_bias=True, bias=False):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = int(expand * d_model)
        self.d_conv = d_conv
        self.conv_bias = conv_bias
        self.bias = bias
        
        if dt_rank == 'auto':
            self.dt_rank = math.ceil(d_model / 16)
        else:
            self.dt_rank = dt_rank

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )

        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        A = repeat(torch.arange(1, self.d_state + 1), 'n -> d n', d=self.d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)

    def forward(self, x):
        (b, l, d) = x.shape

        x_and_res = self.in_proj(x)
        (x, res) = x_and_res.split(split_size=[self.d_inner, self.d_inner], dim=-1)

        x = rearrange(x, 'b l d_in -> b d_in l')
        x = self.conv1d(x)[:, :, :l]
        x = rearrange(x, 'b d_in l -> b l d_in')

        x = F.silu(x)
        y = self.ssm(x)
        y = y * F.silu(res)
        output = self.out_proj(y)
        return output

    def ssm(self, x):
        (d_in, n) = self.A_log.shape

        A = -torch.exp(self.A_log.float())
        D = self.D.float()

        x_dbl = self.x_proj(x)
        (delta, B, C) = x_dbl.split(split_size=[self.dt_rank, n, n], dim=-1)
        delta = F.softplus(self.dt_proj(delta))

        y = self.selective_scan(x, delta, A, B, C, D)
        return y

    def selective_scan(self, u, delta, A, B, C, D):
        (b, l, d_in) = u.shape
        n = A.shape[1]

        deltaA = torch.exp(einsum(delta, A, 'b l d_in, d_in n -> b l d_in n'))
        deltaB_u = einsum(delta, B, u, 'b l d_in, b l n, b l d_in -> b l d_in n')

        x = torch.zeros((b, d_in, n), device=deltaA.device)
        ys = []    
        for i in range(l):
            x = deltaA[:, i] * x + deltaB_u[:, i]
            y = einsum(x, C[:, i, :], 'b d_in n, b n -> b d_in')
            ys.append(y)
        y = torch.stack(ys, dim=1)

        y = y + u * D
        return y

class ResidualBlock(nn.Module):
    def __init__(self, d_model, d_state=16, expand=2):
        super().__init__()
        self.mixer = MambaBlock(d_model=d_model, d_state=d_state, expand=expand)
        self.norm = RMSNorm(d_model)

    def forward(self, x):
        return self.mixer(self.norm(x)) + x

class EEGMamba(nn.Module):
    def __init__(self, n_chans, n_outputs, n_times, d_model=64, n_layers=2, d_state=16, expand=2):
        super().__init__()
        self.conv1 = nn.Conv2d(1, d_model, (n_chans, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(d_model)
        self.elu = nn.ELU()
        
        # Downsample the temporal dimension by 4x to speed up sequence modeling
        self.pool = nn.MaxPool2d((1, 4))
        
        self.mamba_layers = nn.ModuleList([
            ResidualBlock(d_model=d_model, d_state=d_state, expand=expand)
            for _ in range(n_layers)
        ])
        
        self.norm = RMSNorm(d_model)
        self.classifier = nn.Linear(d_model, n_outputs)

    def forward(self, x):
        # Input shape: (batch_size, n_chans, n_times)
        x = x.unsqueeze(1) # shape: (batch_size, 1, n_chans, n_times)
        x = self.elu(self.bn1(self.conv1(x))) # shape: (batch_size, d_model, 1, n_times)
        x = self.pool(x) # shape: (batch_size, d_model, 1, n_times // 4)
        
        x = x.squeeze(2) # shape: (batch_size, d_model, seq_len)
        x = x.permute(0, 2, 1) # shape: (batch_size, seq_len, d_model)
        
        for layer in self.mamba_layers:
            x = layer(x)
            
        x = self.norm(x)
        x = x.mean(dim=1) # shape: (batch_size, d_model)
        logits = self.classifier(x)
        return logits

def create_model(config, n_chans, n_classes, n_times):
    model_name = config['model']['name']
    if model_name == 'EEGNetv4':
        print('Creating EEGNetV4 Model')
        model = bmodels.EEGNetv4(
            n_chans=n_chans,
            n_outputs=n_classes,
            n_times=n_times
        )
    elif model_name == 'EEGMamba':
        print('Creating EEGMamba Model')
        model = EEGMamba(
            n_chans=n_chans,
            n_outputs=n_classes,
            n_times=n_times
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    return model