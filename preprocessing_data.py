import os
import csv
import torch
import torchaudio
from  classes.models.PathologicalFeature.PathologicalFeatureExtractor import PathologicalFeatureExtractor

DB_PATH = "dataset/"
TSV_PATH = "preprocessed_data/features.tsv"
SECONDS = 1                                     # Number of seconds to process
NB_TIME = 16000 * SECONDS 
SEGMENT_STRIDE = NB_TIME // 2                   # 50% overlap

# Make sure the correct backend is used for decoding mp3
torchaudio.set_audio_backend("soundfile")

extractor = PathologicalFeatureExtractor()

rows = []

for label in ["Spoof", "Bonafide"]:
    label_path = os.path.join(DB_PATH, label)
    for model_name in os.listdir(label_path):
        model_path = os.path.join(label_path, model_name)
        if not os.path.isdir(model_path):
            continue

        for file_name in os.listdir(model_path):
            if not file_name.endswith(".wav"):
                continue

            file_path = os.path.normpath(os.path.join(model_path, file_name))

            try:
                waveform, sr = torchaudio.load(file_path)
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)
                waveform = waveform.squeeze(0)  # [T]
                T = waveform.shape[0]

                # Extract patho features
                patho_feat = extractor(file_path).float()  # [30]

                # Segment waveform + concat
                segment_index = 0
                for start in range(0, T - NB_TIME + 1, SEGMENT_STRIDE):
                    segment = waveform[start:start + NB_TIME]
                    combined = torch.cat([segment, patho_feat], dim=0)  # [16030]

                    # Save as row: [feature_list_str, file_name]
                    feature_list = combined.tolist()
                    rows.append({
                        "features": str(feature_list),  # stored as stringified list
                        "path": file_name,
                        "segment": segment_index
                    })

            except Exception as e:
                print(f"[ERROR] {file_path}: {e}")

# Save features
os.makedirs(os.path.dirname(TSV_PATH), exist_ok=True)
with open(TSV_PATH, mode='w', newline='') as tsvfile:
    writer = csv.DictWriter(tsvfile, fieldnames=["features", "path", "segment"], delimiter='\t')
    writer.writeheader()
    writer.writerows(rows)