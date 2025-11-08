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
        # Handle different model types
        if self.is_hf_model:
            # Hugging Face transformers model
            output = self.model(input_tmp, output_hidden_states=True)
            emb = output.last_hidden_state
            # Create layer_results format compatible with fairseq
            layerresult = [(layer.transpose(0, 1), None) for layer in output.hidden_states]
        else:
            # Fairseq model
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
        # Note: fc1 will be created dynamically based on actual feature size
        self.fc1 = None  # Will be initialized on first forward pass
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
        
        # Initialize fc1 dynamically based on actual feature size
        if self.fc1 is None:
            feature_size = x.size(1)
            self.fc1 = nn.Linear(feature_size, 1024).to(x.device)
            print(f"Initialized fc1 with input size: {feature_size}")
        
        # Classification layers
        x = self.fc1(x)
        x = self.selu(x)
        x = self.fc3(x)
        x = self.selu(x)
        
        # Log softmax output
        output = self.logsoftmax(x)
        
        return output
