import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import braindecode.models as bmodels
import pdb
# ==========================================
# 1. Core Mathematical and Norm Foundations
# ==========================================

class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x, z=None):
        if z is not None:
            x = x * silu(z)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

def silu(x):
    return x * torch.sigmoid(x)

def segsum(x: torch.Tensor, device=None) -> torch.Tensor:
    T = x.size(-1)
    x = repeat(x, "... d -> ... d e", e=T)
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=-1)
    x = x.masked_fill(~mask, 0)
    x_segsum = torch.cumsum(x, dim=-2)
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=0)
    x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
    return x_segsum

# ==========================================
# 2. Structured State Space Dual (SSD) Core
# ==========================================

def ssd(x, A, B, C, chunk_size, initial_states=None, device=None):
    assert x.shape[1] % chunk_size == 0
    x, A, B, C = [
        rearrange(m, "b (c l) ... -> b c l ...", l=chunk_size) for m in (x, A, B, C)
    ]

    A = rearrange(A, "b c l h -> b h c l")
    A_cumsum = torch.cumsum(A, dim=-1)

    L = torch.exp(segsum(A, device=device))
    Y_diag = torch.einsum("bclhn, bcshn, bhcls, bcshp -> bclhp", C, B, L, x)

    decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
    states = torch.einsum("bclhn, bhcl, bclhp -> bchpn", B, decay_states, x)

    if initial_states is None:
        initial_states = torch.zeros_like(states[:, :1])
    states = torch.cat([initial_states, states], dim=1)
    decay_chunk = torch.exp(segsum(F.pad(A_cumsum[:, :, :, -1], (1, 0)), device=device))
    new_states = torch.einsum("bhzc, bchpn -> bzhpn", decay_chunk, states)
    states, final_state = new_states[:, :-1], new_states[:, -1]

    state_decay_out = torch.exp(A_cumsum)
    Y_off = torch.einsum("bclhn, bchpn, bhcl -> bclhp", C, states, state_decay_out)

    Y = rearrange(Y_diag + Y_off, "b c l h p -> b (c l) h p")

    return Y, final_state

# ==========================================
# 3. Mamba-2 Layers
# ==========================================

class Mamba2(nn.Module):
    def __init__(self, d_model, d_state=64, headdim=50, expand=2, chunk_size=64):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.headdim = headdim
        self.expand = expand
        self.chunk_size = chunk_size
        self.d_inner = expand * d_model
        self.nheads = self.d_inner // headdim
        
        # Order: (z, x, B, C, dt)
        d_in_proj = 2 * self.d_inner + 2 * d_state + self.nheads
        self.in_proj = nn.Linear(d_model, d_in_proj, bias=False)

        conv_dim = self.d_inner + 2 * d_state
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            kernel_size=4,
            groups=conv_dim,
            padding=3,
        )

        self.dt_bias = nn.Parameter(torch.empty(self.nheads))
        self.A_log = nn.Parameter(torch.empty(self.nheads))
        self.D = nn.Parameter(torch.empty(self.nheads))
        self.norm = RMSNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        
        # Initialize
        nn.init.zeros_(self.dt_bias)
        nn.init.ones_(self.A_log)
        nn.init.ones_(self.D)

    def forward(self, u: torch.Tensor):
        A = -torch.exp(self.A_log)
        zxbcdt = self.in_proj(u)
        z, xBC, dt = torch.split(
            zxbcdt,
            [
                self.d_inner,
                self.d_inner + 2 * self.d_state,
                self.nheads,
            ],
            dim=-1,
        )
        dt = F.softplus(dt + self.dt_bias)

        xBC = silu(
            self.conv1d(xBC.transpose(1, 2)).transpose(1, 2)[:, : u.shape[1], :]
        )
        x, B, C = torch.split(
            xBC, [self.d_inner, self.d_state, self.d_state], dim=-1
        )
        x = rearrange(x, "b l (h p) -> b l h p", p=self.headdim)
        y, ssm_state = ssd(
            x * dt.unsqueeze(-1),
            A * dt,
            rearrange(B, "b l n -> b l 1 n"),
            rearrange(C, "b l n -> b l 1 n"),
            self.chunk_size,
            device=u.device,
        )
        y = y + x * self.D.unsqueeze(-1)
        y = rearrange(y, "b l h p -> b l (h p)")
        y = self.norm(y, z)
        y = self.out_proj(y)

        return y

class Mamba2Layer(nn.Module):
    def __init__(self, d_model, d_state=64, headdim=50, expand=2, chunk_size=64):
        super().__init__()
        self.mixer = Mamba2(d_model=d_model, d_state=d_state, headdim=headdim, expand=expand, chunk_size=chunk_size)
        self.norm = RMSNorm(d_model)

    def forward(self, x):
        return self.mixer(self.norm(x)) + x

class MixerModel(nn.Module):
    def __init__(self, d_model, n_layer=12, d_state=64, headdim=50, expand=2, chunk_size=64):
        super().__init__()
        self.layers = nn.ModuleList([
            Mamba2Layer(d_model=d_model, d_state=d_state, headdim=headdim, expand=expand, chunk_size=chunk_size)
            for _ in range(n_layer)
        ])
        self.norm_f = RMSNorm(d_model)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.norm_f(x)
        return x

# ==========================================
# 4. Front-End Feature Extraction Front-End
# ==========================================

class PatchEmbedding(nn.Module):
    def __init__(self, in_dim, out_dim, d_model, seq_len):
        super().__init__()
        self.d_model = d_model
        self.positional_encoding = nn.Sequential(
            nn.Conv2d(in_channels=d_model, out_channels=d_model, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3),
                      groups=d_model, bias=False),
        )
        self.mask_encoding = nn.Parameter(torch.zeros(in_dim), requires_grad=False)

        self.proj_in = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=25, kernel_size=(1, 49), stride=(1, 25), padding=(0, 24), bias=False),
            nn.GroupNorm(5, 25),
            nn.GELU(),
        )
        self.spectral_proj = nn.Sequential(
            nn.Linear(101, d_model, bias=False),
            nn.Dropout(0.1),
        )

    def forward(self, x, mask=None):
        pdb.set_trace()
        bz, ch_num, patch_num, patch_size = x.shape
        if mask is None:
            mask_x = x
        else:
            mask_x = x.clone()
            mask_x[mask == 1] = self.mask_encoding

        mask_x = rearrange(mask_x, 'b c l d -> b d c l')
        time_x = rearrange(mask_x, 'b d c l -> b (c l) d').unsqueeze(1)

        time_emb = self.proj_in(time_x)
        time_emb = time_emb.permute(0, 2, 1, 3).contiguous().view(bz, ch_num, patch_num, self.d_model)

        freq_x = rearrange(mask_x, 'b d c l -> b c l d')
        spectral = torch.fft.rfft(freq_x, dim=-1, norm='forward')
        spectral = torch.abs(spectral)
        spectral_emb = self.spectral_proj(spectral)
        patch_emb = time_emb + spectral_emb

        positional_embedding = self.positional_encoding(patch_emb.permute(0, 3, 1, 2))
        positional_embedding = positional_embedding.permute(0, 2, 3, 1)

        patch_emb = patch_emb + positional_embedding

        return patch_emb

# ==========================================
# 5. Integrated EEGMamba Framework
# ==========================================

class EEGMamba(nn.Module):
    def __init__(self, n_chans, n_outputs, n_times, d_model=200, n_layer=12, d_state=64, headdim=50, expand=2, chunk_size=64):
        super().__init__()
        self.n_chans = n_chans
        self.n_times = n_times
        self.d_model = d_model
        self.chunk_size = chunk_size
        
        # Calculate patch dimensions
        self.patch_size = 200
        self.patch_num = math.ceil(n_times / self.patch_size)
        self.pad_len = self.patch_num * self.patch_size - n_times
        
        self.patch_embedding = PatchEmbedding(in_dim=self.patch_size, out_dim=d_model, d_model=d_model, seq_len=self.patch_num)
        
        self.encoder = MixerModel(
            d_model=d_model,
            n_layer=n_layer,
            d_state=d_state,
            headdim=headdim,
            expand=expand,
            chunk_size=chunk_size
        )
        
        self.proj_out = nn.Sequential(
            nn.Linear(d_model, d_model)
        )
        
        # FIX: Replaced flattened linear bottleneck with an expressive, deep,
        # non-linear classification head tailored for downstream 90-class margins.
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, n_outputs)
        )

    def forward(self, x):
        pdb.set_trace()
        # Input shape: (batch_size, n_chans, n_times)
        
        # Pad time dimension to a multiple of patch_size (200)
        if self.pad_len > 0:
            x = F.pad(x, (0, self.pad_len))
            
        # Reshape to (batch_size, n_chans, patch_num, patch_size)
        x = x.view(x.shape[0], self.n_chans, self.patch_num, self.patch_size)
        
        # Pass through patch embedding
        hidden_states = self.patch_embedding(x)
        
        # Reshape for Mamba: (batch_size, n_chans * patch_num, d_model)
        hidden_states = rearrange(hidden_states, 'b c l d -> b (c l) d')
        
        # Pad sequence dimension to a multiple of chunk_size (64)
        seq_len = hidden_states.shape[1]
        pad_seq = (self.chunk_size - seq_len % self.chunk_size) % self.chunk_size
        if pad_seq > 0:
            hidden_states = F.pad(hidden_states, (0, 0, 0, pad_seq))
            
        hidden_states = self.encoder(hidden_states)
        
        # Truncate back to original sequence length
        if pad_seq > 0:
            hidden_states = hidden_states[:, :seq_len, :]
            
        # Reshape back to (batch_size, n_chans, patch_num, d_model)
        hidden_states = rearrange(hidden_states, 'b (c l) d -> b c l d', l=self.patch_num)
        
        out = self.proj_out(hidden_states) # (batch, channels, patches, d_model)
        
        # FIX: Swapped global flattening out for a Spatial-Temporal Global Average Pooling.
        # This keeps the feature vector scale stable across input configurations and retains
        # the descriptive integrity of the pretrained representations.
        out = torch.mean(out, dim=[1, 2]) # (batch, d_model)
        
        logits = self.classifier(out)
        return logits

# ==========================================
# 6. Model Generation & Pretrained Initialization
# ==========================================

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
        print('Creating EEGMamba Model with pretrained weights...')
        model = EEGMamba(
            n_chans=n_chans,
            n_outputs=n_classes,
            n_times=n_times
        )
        try:
            from huggingface_hub import hf_hub_download
            print("Downloading pretrained weights from weighting666/EEGMamba...")
            weights_path = hf_hub_download(repo_id="weighting666/EEGMamba", filename="pretrained_EEGMamba.pth")
            print(f"Loading weights from {weights_path}")
            state_dict = torch.load(weights_path, map_location='cpu')
            
            model_dict = model.state_dict()
            
            # FIX: Explicitly exclude 'classifier' keys from weight matches.
            # This ensures your backbone loads pretrained features correctly, while your downstream
            # 90-class classification layers initialize fresh without random matrix collisions.
            pretrained_dict = {
                k: v for k, v in state_dict.items() 
                if k in model_dict and "classifier" not in k and v.shape == model_dict[k].shape
            }
            
            print(f"Successfully matched {len(pretrained_dict)} out of {len(state_dict)} pretrained backbone parameters.")
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            print("Pretrained Mamba-2 backbone loaded successfully! Downstream head configured for fine-tuning.")
        except Exception as e:
            raise RuntimeError(f"Failed to load pretrained EEGMamba weights: {e}")
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    return model