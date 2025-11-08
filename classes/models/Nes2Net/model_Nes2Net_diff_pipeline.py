# -*- encoding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

___author__ = "Adapted from Tianchi Liu"
__email__ = "tianchi_liu@u.nus.edu"


# -------------------------
# Helper Modules
# -------------------------
class SEModule(nn.Module):
    """Squeeze-and-Excitation module"""
    def __init__(self, channels, SE_ratio=8):
        super(SEModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, channels // SE_ratio, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv1d(channels // SE_ratio, channels, kernel_size=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, input):
        x = self.se(input)
        return input * x


class Bottle2neck(nn.Module):
    """Res2Net bottleneck block"""
    def __init__(self, inplanes, planes, kernel_size=None, dilation=None, scale=8, SE_ratio=8):
        super(Bottle2neck, self).__init__()
        width = int(math.floor(planes / scale))
        self.conv1 = nn.Conv1d(inplanes, width * scale, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(width * scale)
        self.nums = scale - 1
        convs = []
        bns = []
        num_pad = math.floor(kernel_size / 2) * dilation
        for i in range(self.nums):
            convs.append(nn.Conv1d(width, width, kernel_size=kernel_size, dilation=dilation, padding=num_pad))
            bns.append(nn.BatchNorm1d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.conv3 = nn.Conv1d(width * scale, planes, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU()
        self.width = width
        self.se = SEModule(planes, SE_ratio)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(sp)
            sp = self.bns[i](sp)
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        out = torch.cat((out, spx[self.nums]), 1)
        out = self.conv3(out)
        out = self.relu(out)
        out = self.bn3(out)
        out = self.se(out)
        out += residual
        return out


class ASTP(nn.Module):
    """Attentive Statistics Pooling
    Channel- and context-dependent statistics pooling, first used in ECAPA-TDNN.
    """
    def __init__(self, in_dim, bottleneck_dim=128, global_context_att=False):
        super(ASTP, self).__init__()
        self.global_context_att = global_context_att

        # Use Conv1d with stride == 1 rather than Linear, then we don't
        # need to transpose inputs.
        if global_context_att:
            self.linear1 = nn.Conv1d(in_dim * 3, bottleneck_dim, kernel_size=1)
        else:
            self.linear1 = nn.Conv1d(in_dim, bottleneck_dim, kernel_size=1)
        self.linear2 = nn.Conv1d(bottleneck_dim, in_dim, kernel_size=1)

    def forward(self, x):
        """
        x: a 3-dimensional tensor in tdnn-based architecture (B,F,T)
            or a 4-dimensional tensor in resnet architecture (B,C,F,T)
            0-dim: batch-dimension, last-dim: time-dimension (frame-dimension)
        """
        if len(x.shape) == 4:
            x = x.reshape(x.shape[0], x.shape[1] * x.shape[2], x.shape[3])
        assert len(x.shape) == 3

        if self.global_context_att:
            context_mean = torch.mean(x, dim=-1, keepdim=True).expand_as(x)
            context_std = torch.sqrt(torch.var(x, dim=-1, keepdim=True) + 1e-10).expand_as(x)
            x_in = torch.cat((x, context_mean, context_std), dim=1)
        else:
            x_in = x

        # DON'T use ReLU here! ReLU may be hard to converge.
        alpha = torch.tanh(self.linear1(x_in))
        alpha = torch.softmax(self.linear2(alpha), dim=2)
        mean = torch.sum(alpha * x, dim=2)
        var = torch.sum(alpha * (x ** 2), dim=2) - mean ** 2
        std = torch.sqrt(var.clamp(min=1e-10))
        return torch.cat([mean, std], dim=1)


# -------------------------
# SSL Model (WavLM)
# -------------------------
class SSLModel(nn.Module):
    """Self-Supervised Learning Model wrapper for WavLM"""
    def __init__(self, device, cp_path='wavlm_large.pt', agg='SEA'):
        super(SSLModel, self).__init__()
        print(f"Loading pretrained WavLM model from: {cp_path}")
        
        try:
            from s3prl import hub
            self.model = getattr(hub, "wavlm_large")()
            print("Successfully loaded WavLM model via s3prl hub")
        except Exception as e:
            print(f"Error loading WavLM model: {e}")
            raise RuntimeError(f"Failed to load WavLM model. Make sure s3prl is installed and the model is available.")
        
        self.device = device
        self.out_dim = 1024
        self.agg = agg
        self.n_layer = 25
        
        # Aggregation modules
        if self.agg == 'SEA':
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.fc_att_merge = nn.Sequential(
                nn.Linear(self.n_layer, int(self.n_layer // 3), bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(int(self.n_layer // 3), self.n_layer, bias=False),
                nn.Sigmoid()
            )
        elif self.agg == 'WeightedSum':
            self.weight_hidd = nn.Parameter(torch.ones(self.n_layer))
        elif self.agg == 'AttM':
            self.n_feat = self.out_dim
            self.W = nn.Parameter(torch.randn(self.n_feat, 1))
            self.W1 = nn.Parameter(torch.randn(self.n_layer, int(self.n_layer // 2)))
            self.W2 = nn.Parameter(torch.randn(int(self.n_layer // 2), self.n_layer))
            self.hidden = int(self.n_layer * self.n_feat / 4)
            self.linear_proj = nn.Linear(self.n_layer * self.n_feat, self.n_feat)
            self.SWISH = nn.SiLU()
        else:
            raise ValueError(f"Unknown aggregation method: {self.agg}")

    def _weighted_sum(self, x):
        feature = x['hidden_states']
        layer_num = len(feature)
        stacked_feature = torch.stack(feature, dim=0)
        _, *origin_shape = stacked_feature.shape
        stacked_feature = stacked_feature.view(layer_num, -1)
        norm_weights = F.softmax(self.weight_hidd[:layer_num], dim=-1)
        weighted_feature = (norm_weights.unsqueeze(-1) * stacked_feature).sum(dim=0)
        weighted_feature = weighted_feature.view(*origin_shape)
        return weighted_feature

    def _SE_merge(self, x):
        feature = x['hidden_states']
        stacked_feature = torch.stack(feature, dim=1)
        b, c, _, _ = stacked_feature.size()
        y = self.avg_pool(stacked_feature).view(b, c)
        y = self.fc_att_merge(y).view(b, c, 1, 1)
        stacked_feature = stacked_feature * y.expand_as(stacked_feature)
        weighted_feature = torch.sum(stacked_feature, dim=1)
        return weighted_feature

    def _Att_merge(self, x):
        x = x['hidden_states']
        x = torch.stack(x, dim=1)
        x_input = x
        x = torch.mean(x, dim=2, keepdim=True)
        x = self.SWISH(torch.matmul(x, self.W))
        x = self.SWISH(torch.matmul(x.view(-1, self.n_layer), self.W1))
        x = torch.sigmoid((torch.matmul(x, self.W2)))
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = torch.mul(x, x_input)
        x = x.permute(0, 2, 3, 1).contiguous().view(x.size(0), x.size(2), -1)
        weighted_feature = self.linear_proj(x)
        return weighted_feature

    def forward(self, input_data):
        input_data = input_data.to(self.device)
        if next(self.model.parameters()).device != input_data.device:
            self.model.to(input_data.device)

        if input_data.ndim == 3:
            input_tmp = input_data[:, :, 0]
        else:
            input_tmp = input_data

        emb = self.model(input_tmp)
        
        if self.agg == 'SEA':
            return self._SE_merge(emb)
        elif self.agg == 'WeightedSum':
            return self._weighted_sum(emb)
        elif self.agg == 'AttM':
            return self._Att_merge(emb)
        else:
            raise ValueError(f"Unknown aggregation method: {self.agg}")


# -------------------------
# Nested Res2Net TDNN (Modified for embedding extraction)
# -------------------------
class Nested_Res2Net_TDNN_Embedding(nn.Module):
    """Nested Res2Net Time-Delay Neural Network (returns embedding before final fc)"""
    def __init__(self, Nes_ratio=[8, 8], input_channel=1024, dilation=2, pool_func='mean', SE_ratio=[8]):
        super(Nested_Res2Net_TDNN_Embedding, self).__init__()
        self.Nes_ratio = Nes_ratio[0]
        assert input_channel % Nes_ratio[0] == 0
        C = input_channel // Nes_ratio[0]
        self.C = C
        
        Build_in_Res2Nets = []
        bns = []
        for i in range(Nes_ratio[0] - 1):
            Build_in_Res2Nets.append(Bottle2neck(C, C, kernel_size=3, dilation=dilation, scale=Nes_ratio[1], SE_ratio=SE_ratio[0]))
            bns.append(nn.BatchNorm1d(C))
        
        self.Build_in_Res2Nets = nn.ModuleList(Build_in_Res2Nets)
        self.bns = nn.ModuleList(bns)
        self.bn = nn.BatchNorm1d(1024)
        self.relu = nn.ReLU()
        self.pool_func = pool_func
        
        # Note: No fc layer here - we return the embedding

    def forward(self, x):
        spx = torch.split(x, self.C, 1)
        for i in range(self.Nes_ratio - 1):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.Build_in_Res2Nets[i](sp)
            sp = self.bns[i](sp)
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        out = torch.cat((out, spx[-1]), 1)
        out = self.bn(out)
        out = self.relu(out)
        
        # Pooling to get embedding
        if self.pool_func == 'mean':
            out = torch.cat([torch.mean(out, dim=2), torch.std(out, dim=2)], dim=1)
        elif self.pool_func == 'ASTP':
            out = ASTP(1024)(out)
        
        # Return embedding (2048-dim for mean pooling)
        return out


# -------------------------
# WavLM-Nes2Net Model (Different Pipeline)
# -------------------------
class WavLMNes2NetDiffPipeline(nn.Module):
    """WavLM-Nes2Net model for audio deepfake detection with pathological features"""
    def __init__(self, model_config, device):
        super().__init__()
        self.device = device
        
        # Extract configuration
        agg = model_config.get('agg', 'SEA')
        Nes_ratio = model_config.get('Nes_ratio', [8, 8])
        dilation = model_config.get('dilation', 2)
        pool_func = model_config.get('pool_func', 'mean')
        SE_ratio = model_config.get('SE_ratio', [8])
        cp_path = model_config.get('cp_path', 'wavlm_large.pt')
        self.fine_tune_ssl = model_config.get('fine_tune_ssl', True)
        self.nb_patho_features = model_config.get('nb_patho_features', 24)
        
        # Create network WavLM
        self.ssl_model = SSLModel(self.device, cp_path=cp_path, agg=agg)
        
        # Backend network (returns embedding instead of classification)
        self.Nested_Res2Net_TDNN = Nested_Res2Net_TDNN_Embedding(
            Nes_ratio=Nes_ratio, 
            input_channel=1024,
            dilation=dilation, 
            pool_func=pool_func, 
            SE_ratio=SE_ratio
        )
        
        # Audio embedding dimension (2048 for mean pooling)
        self.audio_embedding_dim = 2048
        
        # Classification layer (audio embedding + pathological features)
        self.fc_out = nn.Linear(self.audio_embedding_dim + self.nb_patho_features, 2)
        
        print('WavLM-Nes2Net (Different Pipeline) model initialized')

    def forward(self, x, is_test=False):
        """
        Forward pass with different pipeline for audio and pathological features
        
        Args:
            x: Input tensor. Can be:
               - Waveform only: (batch, seq_len)
               - Waveform + patho: (batch, seq_len + nb_patho_features)
            is_test: If True, return only audio embedding (for feature extraction)
        
        Returns:
            If is_test=True: (audio_embedding, None)
            If is_test=False: (output, None)
        """
        # Determine if pathological features are included
        if x.size(1) > 64000:  # Assuming waveform is 64000 samples
            # Split input: waveform and pathological features
            x_waveform = x[:, :64000]  # (batch, 64000)
            x_patho = x[:, 64000:]     # (batch, nb_patho_features)
            has_patho = True
        else:
            # Only waveform
            x_waveform = x
            has_patho = False
        
        x_waveform = x_waveform.to(self.device)
        
        # Pre-trained WavLM model
        if not self.fine_tune_ssl:
            with torch.no_grad():
                x_ssl_feat = self.ssl_model(x_waveform)
        else:
            x_ssl_feat = self.ssl_model(x_waveform)
        
        # x_ssl_feat shape: [batch, 1024, time]
        x_ssl_feat = x_ssl_feat.permute(0, 2, 1)  # [batch, time, 1024] -> [batch, 1024, time]
        
        # Backend processing to get audio embedding
        audio_emb = self.Nested_Res2Net_TDNN(x_ssl_feat)  # [batch, 2048]
        
        # If test mode, return only audio embedding
        if is_test:
            return audio_emb, None
        
        # Combine with pathological features (if available)
        if has_patho:
            x_patho = x_patho.float()
            combined = torch.cat((audio_emb, x_patho), dim=1)  # (batch, 2048 + nb_patho_features)
            
            # Normalize the combined features (similar to other diff_pipeline models)
            normed = combined / (combined.norm(p=2, dim=1, keepdim=True) + 1e-8) * 10
            
            # Final classification
            out = self.fc_out(normed)
        else:
            # If no pathological features, pad with zeros
            x_patho_zeros = torch.zeros(audio_emb.size(0), self.nb_patho_features, device=audio_emb.device)
            combined = torch.cat((audio_emb, x_patho_zeros), dim=1)
            
            # Normalize the combined features
            normed = combined / (combined.norm(p=2, dim=1, keepdim=True) + 1e-8) * 10
            
            # Final classification
            out = self.fc_out(normed)
        
        return out, None  # Return None for second output to match interface


if __name__ == '__main__':
    print("WavLM-Nes2Net (Different Pipeline) model module")
    
    # Example configuration
    model_config = {
        'agg': 'SEA',
        'Nes_ratio': [8, 8],
        'dilation': 2,
        'pool_func': 'mean',
        'SE_ratio': [8],
        'cp_path': 'wavlm_large.pt',
        'fine_tune_ssl': True,
        'nb_patho_features': 24
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = WavLMNes2NetDiffPipeline(model_config, device)
    
    # Test forward pass with combined features
    x = torch.rand((4, 64024)).to(device)  # 64000 waveform + 24 patho features
    model = model.to(device)
    output, _ = model(x)
    print(f"Output shape: {output.shape}")
    
    # Test with waveform only
    x_wav = torch.rand((4, 32000)).to(device)
    output_wav, _ = model(x_wav)
    print(f"Output shape (waveform only): {output_wav.shape}")
    
    # Count parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {trainable_params:,}")
