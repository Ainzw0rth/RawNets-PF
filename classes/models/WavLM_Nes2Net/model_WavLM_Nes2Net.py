# -*- encoding: utf-8 -*-
"""
WavLM-Nes2Net Model Wrapper
Wraps the baseline WavLM_Nes2Net_noRes model for compatibility with training scripts
"""

import sys
import os

# Add parent directory to path to import from models/
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, parent_dir)

# Import the baseline model
from models.WavLM_Nes2Net import WavLM_Nes2Net_noRes


class WavLMNes2Net(WavLM_Nes2Net_noRes):
    """
    Wrapper class for WavLM_Nes2Net_noRes to match the naming convention
    and provide a consistent interface for the training pipeline.
    
    This class inherits from WavLM_Nes2Net_noRes and maintains the same
    architecture and forward pass behavior.
    """
    
    def __init__(self, args, device):
        """
        Initialize WavLM-Nes2Net model
        
        Args:
            args: Namespace containing model configuration:
                - agg: Aggregation method ('SEA', 'AttM', 'WeightedSum')
                - Nes_ratio: Nested ratio for Res2Net blocks
                - dilation: Dilation factor
                - pool_func: Pooling function ('mean', 'ASTP')
                - SE_ratio: SE module downsampling ratio
            device: Device to run the model on ('cuda' or 'cpu')
        """
        super(WavLMNes2Net, self).__init__(args=args, device=device)
        
    def forward(self, x, SSL_freeze=False):
        """
        Forward pass through the model
        
        Args:
            x: Input waveform tensor of shape (batch_size, seq_len) or (batch_size, seq_len, 1)
            SSL_freeze: Whether to freeze SSL model during forward pass
        
        Returns:
            Output tensor of shape (batch_size, 1) for binary classification
        """
        return super(WavLMNes2Net, self).forward(x, SSL_freeze=SSL_freeze)


if __name__ == '__main__':
    import torch
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--agg", type=str, default='SEA', choices=['SEA', 'AttM', 'WeightedSum'],
                        help="the aggregation method for SSL")
    parser.add_argument("--dilation", type=int, default=1, help="dilation")
    parser.add_argument("--pool_func", type=str, default='mean', choices=['mean', 'ASTP'],
                        help="pooling function, choose from mean and ASTP")
    parser.add_argument("--SE_ratio", type=int, nargs='+', default=[1], help="SE downsampling ratio in the bottleneck")
    parser.add_argument("--Nes_ratio", type=int, nargs='+', default=[8, 8], help="Nes_ratio, from outer to inner")

    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = WavLMNes2Net(args=args, device=device)
    
    # Test forward pass
    x = torch.rand((4, 32000)).to(device)
    model = model.to(device)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Output: {y}")
    
    # Count parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal trainable parameters: {trainable_params:,}")
    
    ssl_params = sum(p.numel() for p in model.ssl_model.parameters() if p.requires_grad)
    print(f"SSL model parameters: {ssl_params:,}")
    
    nes2net_params = sum(p.numel() for p in model.Nested_Res2Net_TDNN.parameters() if p.requires_grad)
    print(f"Nested_Res2Net_TDNN parameters: {nes2net_params:,}")
