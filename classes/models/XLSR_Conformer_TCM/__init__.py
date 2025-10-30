"""
XLSR-Conformer with Temporal-Channel Modeling (TCM)
For synthetic speech detection

This package contains:
- Model implementation (XLSRConformerTCM)
- Training utilities
- Evaluation utilities
"""

from .model_XLSR_Conformer_TCM import XLSRConformerTCM, SSLModel, MyConformer
from .trainer_XLSR_Conformer_TCM import (
    train_xlsr_conformer_tcm_with_loaders,
    validate_xlsr_conformer_tcm,
    test_xlsr_conformer_tcm,
    test_xlsr_conformer_tcm_with_loaders,
    save_model_xlsr_conformer_tcm,
    load_model_xlsr_conformer_tcm,
    produce_evaluation_file
)

__all__ = [
    'XLSRConformerTCM',
    'SSLModel',
    'MyConformer',
    'train_xlsr_conformer_tcm_with_loaders',
    'validate_xlsr_conformer_tcm',
    'test_xlsr_conformer_tcm',
    'test_xlsr_conformer_tcm_with_loaders',
    'save_model_xlsr_conformer_tcm',
    'load_model_xlsr_conformer_tcm',
    'produce_evaluation_file'
]
