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
# XLSR-SLS Model with Pathological Features (Different Pipeline)
# -------------------------
class XLSRSLSDiffPipeline(nn.Module):
    def __init__(self, model_config, device):
        super().__init__()
        self.device = device
        
        # Extract configuration
        cp_path = model_config.get('cp_path', 'xlsr2_300m.pt')
        self.fine_tune_ssl = model_config.get('fine_tune_ssl', True)
        self.nb_patho_features = model_config.get('nb_patho_features', 24)
        
        # Create SSL model (XLSR)
        self.ssl_model = SSLModel(self.device, cp_path=cp_path)
        
        # Batch normalization for feature maps
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.selu = nn.SELU(inplace=True)
        
        # Attention layer for selective layer scoring
        self.fc0 = nn.Linear(1024, 1)  # Layer attention weights
        self.sig = nn.Sigmoid()
        
        # Audio embedding layer
        # Note: fc1 will be created dynamically based on actual feature size
        self.fc1 = None  # Will be initialized on first forward pass
        self.audio_embedding_dim = 1024
        
        # Classification layers (audio embedding + pathological features)
        self.fc2 = nn.Linear(self.audio_embedding_dim + self.nb_patho_features, 2)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        
        # Set SSL model training mode
        if not self.fine_tune_ssl:
            for param in self.ssl_model.parameters():
                param.requires_grad = False
            print("SSL model frozen (fine_tune_ssl=False)")
        else:
            print("SSL model will be fine-tuned (fine_tune_ssl=True)")

    def forward(self, x, is_test=False):
        """
        Forward pass with different pipeline for audio and pathological features
        
        Args:
            x: Input tensor. Can be:
               - Waveform only: (batch, seq_len)
               - Waveform + patho: (batch, seq_len + nb_patho_features)
            is_test: If True, return only audio embedding (for feature extraction)
        
        Returns:
            If is_test=True: audio embedding (batch, audio_embedding_dim)
            If is_test=False: log softmax output (batch, 2)
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
        
        # Extract SSL features and layer results from waveform
        # x_waveform shape: (batch, seq_len)
        x_ssl_feat, layerResult = self.ssl_model.extract_feat(x_waveform.squeeze(-1) if x_waveform.ndim == 3 else x_waveform)
        
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
        audio_feat = self.first_bn(fullfeature)
        audio_feat = self.selu(audio_feat)
        
        # Max pooling
        audio_feat = F.max_pool2d(audio_feat, (3, 3))
        
        # Flatten for classification
        audio_feat = torch.flatten(audio_feat, 1)
        
        # Initialize fc1 dynamically based on actual feature size
        if self.fc1 is None:
            feature_size = audio_feat.size(1)
            self.fc1 = nn.Linear(feature_size, self.audio_embedding_dim).to(audio_feat.device)
            print(f"Initialized fc1 with input size: {feature_size} -> {self.audio_embedding_dim}")
        
        # Get audio embedding
        audio_emb = self.fc1(audio_feat)
        audio_emb = self.selu(audio_emb)
        
        # If test mode, return only audio embedding
        if is_test:
            return audio_emb
        
        # Combine with pathological features (if available)
        if has_patho:
            x_patho = x_patho.float()
            combined = torch.cat((audio_emb, x_patho), dim=1)  # (batch, audio_embedding_dim + nb_patho_features)
            
            # Normalize the combined features (similar to RawNet2_diff_pipeline)
            normed = combined / (combined.norm(p=2, dim=1, keepdim=True) + 1e-8) * 10
            
            # Final classification
            out = self.fc2(normed)
        else:
            # If no pathological features, pad with zeros
            x_patho_zeros = torch.zeros(audio_emb.size(0), self.nb_patho_features, device=audio_emb.device)
            combined = torch.cat((audio_emb, x_patho_zeros), dim=1)
            
            # Normalize the combined features
            normed = combined / (combined.norm(p=2, dim=1, keepdim=True) + 1e-8) * 10
            
            # Final classification
            out = self.fc2(normed)
        
        # Log softmax output
        output = self.logsoftmax(out)
        
        return output
