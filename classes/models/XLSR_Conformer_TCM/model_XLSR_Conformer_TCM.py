# -*- encoding: utf-8 -*-

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from einops import rearrange
from einops.layers.torch import Rearrange
import fairseq
from torch.nn.modules.transformer import _get_clones
from torch import Tensor

# -------------------------
# Helper Functions
# -------------------------
def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def calc_same_padding(kernel_size):
    pad = kernel_size // 2
    return (pad, pad - (kernel_size + 1) % 2)

def sinusoidal_embedding(n_channels, dim):
    pe = torch.FloatTensor([[p / (10000 ** (2 * (i // 2) / dim)) for i in range(dim)]
                            for p in range(n_channels)])
    pe[:, 0::2] = torch.sin(pe[:, 0::2])
    pe[:, 1::2] = torch.cos(pe[:, 1::2])
    return pe.unsqueeze(0)

# -------------------------
# Helper Classes
# -------------------------
class Swish(nn.Module):
    def forward(self, x):
        return x * x.sigmoid()

class GLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        out, gate = x.chunk(2, dim=self.dim)
        return out * gate.sigmoid()

class DepthWiseConv1d(nn.Module):
    def __init__(self, chan_in, chan_out, kernel_size, padding):
        super().__init__()
        self.padding = padding
        self.conv = nn.Conv1d(chan_in, chan_out, kernel_size, groups=chan_in)

    def forward(self, x):
        x = F.pad(x, self.padding)
        return self.conv(x)

# -------------------------
# Attention and Feedforward Modules
# -------------------------
class Scale(nn.Module):
    def __init__(self, scale, fn):
        super().__init__()
        self.fn = fn
        self.scale = scale

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) * self.scale

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

class Attention(nn.Module):
    # Head Token attention: https://arxiv.org/pdf/2210.05958.pdf
    def __init__(self, dim, heads=8, dim_head=64, qkv_bias=False, dropout=0., proj_drop=0.):
        super().__init__()
        self.num_heads = heads
        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5

        self.qkv = nn.Linear(dim, inner_dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(inner_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.act = nn.GELU()
        self.ht_proj = nn.Linear(dim_head, dim, bias=True)
        self.ht_norm = nn.LayerNorm(dim_head)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_heads, dim))
    
    def forward(self, x, mask=None):
        B, N, C = x.shape

        # head token
        head_pos = self.pos_embed.expand(x.shape[0], -1, -1)
        x_ = x.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) 
        x_ = x_.mean(dim=2)  # now the shape is [B, h, 1, d//h]
        x_ = self.ht_proj(x_).reshape(B, -1, self.num_heads, C // self.num_heads)
        x_ = self.act(self.ht_norm(x_)).flatten(2)
        x_ = x_ + head_pos
        x = torch.cat([x, x_], dim=1)
        
        # normal mhsa
        qkv = self.qkv(x).reshape(B, N+self.num_heads, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N+self.num_heads, C)
        x = self.proj(x)
        
        # merge head tokens into cls token
        cls, patch, ht = torch.split(x, [1, N-1, self.num_heads], dim=1)
        cls = cls + torch.mean(ht, dim=1, keepdim=True) + torch.mean(patch, dim=1, keepdim=True)
        x = torch.cat([cls, patch], dim=1)

        x = self.proj_drop(x)

        return x, attn

class FeedForward(nn.Module):
    def __init__(
        self,
        dim,
        mult=4,
        dropout=0.
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class ConformerConvModule(nn.Module):
    def __init__(
        self,
        dim,
        causal=False,
        expansion_factor=2,
        kernel_size=31,
        dropout=0.
    ):
        super().__init__()

        inner_dim = dim * expansion_factor
        padding = calc_same_padding(kernel_size) if not causal else (kernel_size - 1, 0)

        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b n c -> b c n'),
            nn.Conv1d(dim, inner_dim * 2, 1),
            GLU(dim=1),
            DepthWiseConv1d(inner_dim, inner_dim, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(inner_dim) if not causal else nn.Identity(),
            Swish(),
            nn.Conv1d(inner_dim, dim, 1),
            Rearrange('b c n -> b n c'),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

# -------------------------
# Conformer Block
# -------------------------
class ConformerBlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head=64,
        heads=8,
        ff_mult=4,
        conv_expansion_factor=2,
        conv_kernel_size=31,
        attn_dropout=0.,
        ff_dropout=0.,
        conv_dropout=0.,
        conv_causal=False
    ):
        super().__init__()
        self.ff1 = FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)
        self.attn = Attention(dim=dim, dim_head=dim_head, heads=heads, dropout=attn_dropout)
        self.conv = ConformerConvModule(dim=dim, causal=conv_causal, expansion_factor=conv_expansion_factor, 
                                        kernel_size=conv_kernel_size, dropout=conv_dropout)
        self.ff2 = FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)

        self.attn = PreNorm(dim, self.attn)
        self.ff1 = Scale(0.5, PreNorm(dim, self.ff1))
        self.ff2 = Scale(0.5, PreNorm(dim, self.ff2))

        self.post_norm = nn.LayerNorm(dim)

    def forward(self, x, mask=None):
        x = self.ff1(x) + x
        attn_x, attn_weight = self.attn(x, mask=mask)
        x = attn_x + x
        x = self.conv(x) + x
        x = self.ff2(x) + x
        x = self.post_norm(x)
        return x, attn_weight

# -------------------------
# Conformer Model
# -------------------------
class MyConformer(nn.Module):
    def __init__(self, emb_size=128, heads=4, ffmult=4, exp_fac=2, kernel_size=16, n_encoders=1):
        super(MyConformer, self).__init__()
        self.dim_head = int(emb_size / heads)
        self.dim = emb_size
        self.heads = heads
        self.kernel_size = kernel_size
        self.n_encoders = n_encoders
        self.positional_emb = nn.Parameter(sinusoidal_embedding(10000, emb_size), requires_grad=False)
        self.encoder_blocks = _get_clones(
            ConformerBlock(
                dim=emb_size, 
                dim_head=self.dim_head, 
                heads=heads, 
                ff_mult=ffmult, 
                conv_expansion_factor=exp_fac, 
                conv_kernel_size=kernel_size
            ),
            n_encoders
        )
        self.class_token = nn.Parameter(torch.rand(1, emb_size))
        self.fc5 = nn.Linear(emb_size, 2)

    def forward(self, x, device):  # x shape [bs, tiempo, frecuencia]
        x = x + self.positional_emb[:, :x.size(1), :]
        x = torch.stack([torch.vstack((self.class_token, x[i])) for i in range(len(x))])  # [bs,1+tiempo,emb_size]
        list_attn_weight = []
        for layer in self.encoder_blocks:
            x, attn_weight = layer(x)  # [bs,1+tiempo,emb_size]
            list_attn_weight.append(attn_weight)
        embedding = x[:, 0, :]  # [bs, emb_size]
        out = self.fc5(embedding)  # [bs,2]
        return out, list_attn_weight

# -------------------------
# SSL Model (XLSR)
# -------------------------
class SSLModel(nn.Module):  # W2V
    def __init__(self, device, cp_path='xlsr2_300m.pt'):
        super(SSLModel, self).__init__()
        print(f"Loading pretrained model from: {cp_path}")
        
        # Load checkpoint
        checkpoint = torch.load(cp_path, map_location='cpu')
        print(f"Checkpoint keys: {checkpoint.keys()}")
        
        # Try multiple loading strategies
        model = None
        
        # Strategy 1: Standard fairseq loading
        try:
            models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
            model = models[0]
            print("✓ Successfully loaded with load_model_ensemble_and_task")
        except Exception as e1:
            print(f"✗ Strategy 1 failed: {type(e1).__name__}")
            
            # Strategy 2: Direct model loading with fixed args
            try:
                print("Trying Strategy 2: Direct model construction...")
                from fairseq.models.wav2vec import Wav2Vec2Model
                
                # Get config from checkpoint
                if 'cfg' in checkpoint and checkpoint['cfg'] is not None:
                    cfg = checkpoint['cfg']
                    # Convert OmegaConf to namespace if needed
                    if hasattr(cfg, 'model'):
                        model_cfg = cfg.model
                    else:
                        model_cfg = cfg
                    
                    # Build model from config
                    model = Wav2Vec2Model.build_model(model_cfg, task=None)
                    model.load_state_dict(checkpoint['model'], strict=False)
                    print("✓ Successfully loaded with Strategy 2")
                else:
                    raise ValueError("No valid cfg in checkpoint")
                    
            except Exception as e2:
                print(f"✗ Strategy 2 failed: {type(e2).__name__}")
                
                # Strategy 3: Load using transformers library as fallback
                try:
                    print("Trying Strategy 3: Using transformers library...")
                    from transformers import Wav2Vec2Model as HFWav2Vec2Model
                    
                    # Try to load with huggingface transformers
                    model = HFWav2Vec2Model.from_pretrained("facebook/wav2vec2-xls-r-300m")
                    print("✓ Successfully loaded with transformers library")
                    print("Note: Using Hugging Face model instead of local checkpoint")
                    
                except Exception as e3:
                    print(f"✗ Strategy 3 failed: {type(e3).__name__}")
                    
                    # All strategies failed
                    print("\n" + "="*60)
                    print("ERROR: All loading strategies failed!")
                    print("="*60)
                    print("\nPlease try one of these solutions:")
                    print("\n1. Download the correct XLSR checkpoint:")
                    print("   wget https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr_53_56k.pt -O xlsr2_300m.pt")
                    print("\n2. Or install transformers and use Hugging Face model:")
                    print("   pip install transformers")
                    print("\n3. Or verify your checkpoint file is not corrupted:")
                    print("   ls -lh xlsr2_300m.pt")
                    raise RuntimeError("Failed to load XLSR model with all strategies")
        
        if model is None:
            raise RuntimeError("Model loading failed - no model was created")
            
        self.model = model
        self.device = device
        self.out_dim = 1024
        
        # Detect which model type we're using (fairseq vs transformers)
        self.is_hf_model = hasattr(model, 'config') and hasattr(model.config, 'model_type')
        if self.is_hf_model:
            print("Detected Hugging Face Transformers model")
        else:
            print("Detected fairseq model")
        
        return

    def extract_feat(self, input_data):
        # put the model to GPU if it not there
        if next(self.model.parameters()).device != input_data.device \
           or next(self.model.parameters()).dtype != input_data.dtype:
            self.model.to(input_data.device, dtype=input_data.dtype)
            self.model.train()      

        # input should be in shape (batch, length)
        if input_data.ndim == 3:
            input_tmp = input_data[:, :, 0]
        else:
            input_tmp = input_data
        
        # [batch, length, dim]
        # Handle different model types
        if self.is_hf_model:
            # Hugging Face Transformers API
            outputs = self.model(input_tmp, output_hidden_states=True)
            emb = outputs.last_hidden_state
        else:
            # fairseq API
            emb = self.model(input_tmp, mask=False, features_only=True)['x']
        
        return emb

# -------------------------
# XLSR-Conformer with TCM Model
# -------------------------
class XLSRConformerTCM(nn.Module):
    def __init__(self, model_config, device):
        super().__init__()
        self.device = device
        
        # Extract configuration
        emb_size = model_config.get('emb_size', 144)
        num_encoders = model_config.get('num_encoders', 4)
        heads = model_config.get('heads', 4)
        kernel_size = model_config.get('kernel_size', 31)
        cp_path = model_config.get('cp_path', 'xlsr2_300m.pt')
        self.fine_tune_ssl = model_config.get('fine_tune_ssl', True)
        
        # Create network wav2vec 2.0
        self.ssl_model = SSLModel(self.device, cp_path=cp_path)
        self.LL = nn.Linear(1024, emb_size)
        
        print('XLSR-Conformer with TCM')
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.selu = nn.SELU(inplace=True)
        self.conformer = MyConformer(
            emb_size=emb_size, 
            n_encoders=num_encoders,
            heads=heads, 
            kernel_size=kernel_size
        )
        
        # Set fine-tuning
        if not self.fine_tune_ssl:
            for param in self.ssl_model.parameters():
                param.requires_grad = False
    
    def forward(self, x):
        # Pre-trained Wav2vec model fine tuning
        x_ssl_feat = self.ssl_model.extract_feat(x.squeeze(-1))
        x = self.LL(x_ssl_feat)  # (bs, frame_number, feat_out_dim) (bs, 208, emb_size)
        x = x.unsqueeze(dim=1)  # add channel #(bs, 1, frame_number, emb_size)
        x = self.first_bn(x)
        x = self.selu(x)
        x = x.squeeze(dim=1)
        out, attn_score = self.conformer(x, self.device)
        return out, attn_score
