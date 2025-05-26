import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import parselmouth
from parselmouth.praat import call
from opendbm import VerbalAcoustics

# -------------------------
# Pathology Extraction Modules
# -------------------------

class PathologicalFeatureExtractor():
    def __init__(self, sr=16000, frame_length=0.5, hop_length=0.25):
        super(PathologicalFeatureExtractor, self).__init__()
        self.sr = sr
        self.frame_len = int(frame_length * sr)
        self.hop_len = int(hop_length * sr)

    def extract_from_file(self, file_path: str) -> torch.Tensor:
        sound = parselmouth.Sound(file_path)
        point_process = call(sound, "To PointProcess (periodic, cc)", 75, 500)

        # Jitter
        # using {local, ppq5, ddp} instead of {local, ppq3, ppq5}, since only the latter is available in Praat and the key of the paper is not about using ppq3, but instead combining multiple pathological features, so ppq3 is interchangeable
        jitter_local = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        jitter_ppq5 = call(point_process, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
        jitter_ddp = call(point_process, "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3)

        # Shimmer
        shimmer_local = call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        shimmer_apq3 = call([sound, point_process], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        shimmer_apq5 = call([sound, point_process], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        shimmer_apq11 = call([sound, point_process], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

        # Harmonicity
        harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        chnr = call(harmonicity, "Get mean", 0, 0)
        nne = call(harmonicity, "Get standard deviation", 0, 0)

        # GNE using OpenDBM, using average of glottal noise frames for the whole audio, just like replacing ppq3, the main point is to combine multiple pathological features
        va_model = VerbalAcoustics()
        va_model.fit(file_path)
        gne_frames = va_model.get_glottal_noise()
        gne_mean = gne_frames.mean().item()

        features = [
            jitter_local, jitter_ppq5, jitter_ddp,
            shimmer_local, shimmer_apq3, shimmer_apq5, shimmer_apq11,
            chnr, nne, gne_mean
        ]

        return torch.tensor(features, dtype=torch.float32)

    def compute_deltas(self, feature_tensor: torch.Tensor) -> torch.Tensor:
        # feature_tensor shape: [B, 10]
        delta = F.pad(feature_tensor[:, 1:] - feature_tensor[:, :-1], (1, 0), mode='replicate')
        delta2 = F.pad(delta[:, 1:] - delta[:, :-1], (1, 0), mode='replicate')
        return delta, delta2

    def __call__(self, file_paths):
        single_input = False
        if isinstance(file_paths, str):
            file_paths = [file_paths]
            single_input = True

        base_features = [self.extract_from_file(p) for p in file_paths]  # list of [10]
        base_tensor = torch.stack(base_features)  # [B, 10]

        delta, delta2 = self.compute_deltas(base_tensor)
        full_features = torch.cat([base_tensor, delta, delta2], dim=1)  # [B, 30]

        return full_features[0] if single_input else full_features