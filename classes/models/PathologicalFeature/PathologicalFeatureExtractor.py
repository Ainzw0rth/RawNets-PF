import torch
import torch.nn.functional as F
import numpy as np
import tempfile
import os
import parselmouth
from parselmouth.praat import call
from opendbm import VerbalAcoustics
from pydub import AudioSegment

# -------------------------
# Pathology Extraction Modules
# -------------------------

class PathologicalFeatureExtractor():
    def __init__(self, sr=16000, frame_length=0.5, hop_length=0.25):
        super(PathologicalFeatureExtractor, self).__init__()
        self.sr = sr
        self.frame_len = int(frame_length * sr)
        self.hop_len = int(hop_length * sr)

    def _convert_to_pcm16(self, file_path: str) -> (str, bool):
        """Convert input file to 16-bit PCM WAV format if needed. Returns new file path and a flag for cleanup."""
        sound = AudioSegment.from_file(file_path)
        if sound.sample_width != 2 or sound.frame_rate != self.sr or sound.channels != 1:
            sound = sound.set_sample_width(2).set_frame_rate(self.sr).set_channels(1)
            temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            sound.export(temp_file.name, format="wav")
            return temp_file.name, True  # True means this file needs cleanup
        return file_path, False
    
    def extract_from_file(self, file_path: str) -> torch.Tensor:
        file_path, is_temp = self._convert_to_pcm16(file_path)

        try:
            sound = parselmouth.Sound(file_path)
            point_process = call(sound, "To PointProcess (periodic, cc)", 75, 500)

            # Jitter
            # jitter (rap) is the equivalent of jitter (ppq3) in Praat, both are used interchangeably
            jitter_local = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
            jitter_ppq3 = call(point_process, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
            jitter_ppq5 = call(point_process, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)

            # Shimmer
            shimmer_local = call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            shimmer_apq3 = call([sound, point_process], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            shimmer_apq5 = call([sound, point_process], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            shimmer_apq11 = call([sound, point_process], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

            # GNE using OpenDBM, using average of glottal noise frames for the whole audio, just like replacing ppq3, the main point is to combine multiple pathological features
            va_model = VerbalAcoustics()
            va_model.fit(file_path)
            gne_frames = va_model.get_glottal_noise()
            gne_mean = gne_frames.mean().item()

            features = [
                jitter_local, jitter_ppq3, jitter_ppq5,
                shimmer_local, shimmer_apq3, shimmer_apq5, shimmer_apq11,
                gne_mean
            ]

            return torch.tensor(features, dtype=torch.float32)

        finally:
            if is_temp and os.path.exists(file_path):
                os.remove(file_path)

    def compute_deltas(self, feature_tensor: torch.Tensor) -> torch.Tensor:
        # feature_tensor shape: [B, 8]
        delta = F.pad(feature_tensor[:, 1:] - feature_tensor[:, :-1], (1, 0), mode='replicate')
        delta2 = F.pad(delta[:, 1:] - delta[:, :-1], (1, 0), mode='replicate')
        return delta, delta2

    def __call__(self, file_paths):
        single_input = False
        if isinstance(file_paths, str):
            file_paths = [file_paths]
            single_input = True

        base_features = [self.extract_from_file(p) for p in file_paths]  # list of [8]
        base_tensor = torch.stack(base_features)  # [B, 8]

        delta, delta2 = self.compute_deltas(base_tensor)
        full_features = torch.cat([base_tensor, delta, delta2], dim=1)  # [B, 8 + 8 + 8]

        return full_features[0] if single_input else full_features