# -*- encoding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import fairseq

# -------------------------
# SSL Model (XLSR)
# -------------------------
class SSLModel(nn.Module):
    def __init__(self, device, cp_path='xlsr2_300m.pt'):
        super(SSLModel, self).__init__()
        print(f"Loading pretrained XLSR model from: {cp_path}")
        
        try:
            # Load checkpoint using fairseq
            model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
            self.model = model[0]
            self.device = device
            self.out_dim = 1024
            print("Successfully loaded XLSR model using fairseq")
        except Exception as e:
            print(f"Error loading XLSR model: {e}")
            raise RuntimeError(f"Failed to load XLSR model from {cp_path}. Make sure the checkpoint file exists and fairseq is installed.")
        
        return

    def extract_feat(self, input_data):
        # Put the model to GPU if it's not there
        if next(self.model.parameters()).device != input_data.device \
           or next(self.model.parameters()).dtype != input_data.dtype:
            self.model.to(input_data.device, dtype=input_data.dtype)
            self.model.train()

        # Input should be in shape (batch, length)
        if input_data.ndim == 3:
            input_tmp = input_data[:, :, 0]
        else:
            input_tmp = input_data
            
        # [batch, length, dim] - Extract features and layer results
        emb = self.model(input_tmp, mask=False, features_only=True)['x']
        layerresult = self.model(input_tmp, mask=False, features_only=True)['layer_results']
        
        return emb, layerresult


# -------------------------
# Attention Feature Extraction
# -------------------------
def getAttenF(layerResult):
    """
    Extract attention-weighted features from layer results
    
    Args:
        layerResult: List of layer outputs from XLSR model
    
    Returns:
        layery: Attention weights [batch, num_layers, 1024]
        fullfeature: Full features [batch, num_layers, seq_len, 1024]
    """
    poollayerResult = []
    fullf = []
    
    for layer in layerResult:
        # layer[0] shape: (seq_len, batch, 1024)
        layery = layer[0].transpose(0, 1).transpose(1, 2)  # (batch, 1024, seq_len)
        layery = F.adaptive_avg_pool1d(layery, 1)  # (batch, 1024, 1)
        layery = layery.transpose(1, 2)  # (batch, 1, 1024)
        poollayerResult.append(layery)

        x = layer[0].transpose(0, 1)  # (batch, seq_len, 1024)
        x = x.view(x.size(0), -1, x.size(1), x.size(2))  # (batch, 1, seq_len, 1024)
        fullf.append(x)

    layery = torch.cat(poollayerResult, dim=1)  # (batch, num_layers, 1024)
    fullfeature = torch.cat(fullf, dim=1)  # (batch, num_layers, seq_len, 1024)
    
    return layery, fullfeature


# -------------------------
# XLSR-SLS Model (Selective Layer Scoring)
# -------------------------
class XLSRSLS(nn.Module):
    def __init__(self, model_config, device):
        super().__init__()
        self.device = device
        
        # Extract configuration
        cp_path = model_config.get('cp_path', 'xlsr2_300m.pt')
        self.fine_tune_ssl = model_config.get('fine_tune_ssl', True)
        
        # Create SSL model (XLSR)
        self.ssl_model = SSLModel(self.device, cp_path=cp_path)
        
        # Batch normalization for feature maps
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.selu = nn.SELU(inplace=True)
        
        # Attention layer for selective layer scoring
        self.fc0 = nn.Linear(1024, 1)  # Layer attention weights
        self.sig = nn.Sigmoid()
        
        # Classification layers
        # Note: The input dimension (22847) is calculated based on the feature map size
        # after pooling. This may need adjustment based on input audio length.
        self.fc1 = nn.Linear(22847, 1024)
        self.fc3 = nn.Linear(1024, 2)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        
        # Set SSL model training mode
        if not self.fine_tune_ssl:
            for param in self.ssl_model.parameters():
                param.requires_grad = False
            print("SSL model frozen (fine_tune_ssl=False)")
        else:
            print("SSL model will be fine-tuned (fine_tune_ssl=True)")

    def forward(self, x):
        # Extract SSL features and layer results
        # x shape: (batch, seq_len) or (batch, seq_len, 1)
        x_ssl_feat, layerResult = self.ssl_model.extract_feat(x.squeeze(-1))
        
        # Get attention features from all layers
        # y0: (batch, num_layers, 1024) - attention weights for each layer
        # fullfeature: (batch, num_layers, seq_len, 1024) - features from all layers
        y0, fullfeature = getAttenF(layerResult)
        
        # Calculate attention weights for selective layer scoring
        y0 = self.fc0(y0)  # (batch, num_layers, 1)
        y0 = self.sig(y0)  # Sigmoid activation
        y0 = y0.view(y0.shape[0], y0.shape[1], y0.shape[2], -1)  # (batch, num_layers, 1, 1)
        
        # Apply attention weights to features
        fullfeature = fullfeature * y0  # (batch, num_layers, seq_len, 1024)
        fullfeature = torch.sum(fullfeature, 1)  # Sum across layers: (batch, seq_len, 1024)
        fullfeature = fullfeature.unsqueeze(dim=1)  # (batch, 1, seq_len, 1024)
        
        # Apply batch normalization
        x = self.first_bn(fullfeature)
        x = self.selu(x)
        
        # Max pooling
        x = F.max_pool2d(x, (3, 3))
        
        # Flatten for classification
        x = torch.flatten(x, 1)
        
        # Classification layers
        x = self.fc1(x)
        x = self.selu(x)
        x = self.fc3(x)
        x = self.selu(x)
        
        # Log softmax output
        output = self.logsoftmax(x)
        
        return output
