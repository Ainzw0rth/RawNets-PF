import os
import torch
import torchaudio
import numpy as np
from classes.models.PathologicalFeature.PathologicalFeatureExtractor import PathologicalFeatureExtractor

DB_PATH = "dataset/"
PREPROC_PATH = "preprocessed_data/"  # Where preprocessed .npy files go
SECONDS = 1
NB_TIME = 16000 * SECONDS
SEGMENT_STRIDE = NB_TIME // 2

torchaudio.set_audio_backend("soundfile")

extractor = PathologicalFeatureExtractor()

for label in ["Spoof", "Bonafide"]:
    label_path = os.path.join(DB_PATH, label)
    for root, dirs, files in os.walk(label_path):
        for file_name in files:
            if not file_name.endswith(".wav"):
                continue

            file_path = os.path.join(root, file_name)
            try:
                waveform, sr = torchaudio.load(file_path)
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)
                waveform = waveform.squeeze(0)
                T = waveform.shape[0]

                # Extract pathological features
                print(f"======================= Extracting features from {file_path} =======================")
                patho_feat = extractor.extract_from_file(file_path).float()  # [30]

                # Compute relative path for saving
                rel_path = os.path.relpath(root, DB_PATH)
                save_dir = os.path.join(PREPROC_PATH, rel_path)
                os.makedirs(save_dir, exist_ok=True)

                # Generate segments and save as .npy
                segment_index = 0
                for start in range(0, T - NB_TIME + 1, SEGMENT_STRIDE):
                    segment = waveform[start:start + NB_TIME]
                    combined = torch.cat([segment, patho_feat], dim=0).numpy()  # Convert to numpy

                    base_name = os.path.splitext(file_name)[0]
                    save_name = f"{base_name}_{segment_index}.npy"
                    save_path = os.path.join(save_dir, save_name)

                    np.save(save_path, combined)
                    segment_index += 1

            except Exception as e:
                print(f"[ERROR] {file_path}: {e}")