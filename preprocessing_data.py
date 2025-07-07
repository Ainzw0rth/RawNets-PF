import os
import sys
import time
import logging
import torchaudio
import numpy as np
from utils.Logger import Logger
from datetime import datetime
from classes.models.PathologicalFeature.PathologicalFeatureExtractor import PathologicalFeatureExtractor

DB_PATH = "dataset/"
PREPROC_PATH = "preprocessed_data/"
SECONDS = 4                             # Length of each segment in seconds  
NB_TIME = 16000 * SECONDS
SEGMENT_STRIDE = NB_TIME // 4           # 25% overlap (1 second overlap for 4 seconds)

torchaudio.set_audio_backend("soundfile")

if __name__ == "__main__":
    # Logger setup
    os.makedirs("logs/preprocessing/", exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"logs/preprocessing/preprocessing_log_{timestamp}.txt"
    log_file = open(log_filename, "w")
    sys.stdout = Logger(sys.stdout, log_file)
    sys.stderr = Logger(sys.stderr, log_file)

    total_files = 0
    total_segments = 0
    total_errors = 0

    extractor = PathologicalFeatureExtractor()

    start_time = time.time()
    print("==================== PREPROCESSING STARTED ====================\n")
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
                    print(f">>> Processing file: {file_path}")
                    patho_feat = extractor(file_path).float() 

                    # Compute relative path
                    rel_path = os.path.relpath(root, DB_PATH)

                    # Define base save dirs per category
                    waveform_dir = os.path.join(PREPROC_PATH, "waveform", rel_path)
                    patho_dir = os.path.join(PREPROC_PATH, "patho", rel_path)
                    combined_dir = os.path.join(PREPROC_PATH, "combined", rel_path)

                    # Create directories
                    os.makedirs(waveform_dir, exist_ok=True)
                    os.makedirs(patho_dir, exist_ok=True)
                    os.makedirs(combined_dir, exist_ok=True)

                    # Generate segments and save as .npy
                    segment_index = 0
                    for start in range(0, T - NB_TIME + 1, SEGMENT_STRIDE):
                        segment = waveform[start:start + NB_TIME]

                        # Convert both to numpy
                        segment_np = segment.numpy()
                        patho_feat_np = patho_feat.numpy()
                        combined = np.concatenate([segment_np, patho_feat_np], axis=0)

                        # Base file naming
                        base_name = os.path.splitext(file_name)[0]
                        save_base = f"{base_name}_{segment_index}"
                        
                        # Save paths (in subfolders)
                        waveform_path = os.path.join(waveform_dir, f"{save_base}.npy")
                        patho_path = os.path.join(patho_dir, f"{save_base}.npy")
                        combined_path = os.path.join(combined_dir, f"{save_base}.npy")

                        # Save the files
                        np.save(waveform_path, segment_np)
                        np.save(patho_path, patho_feat_np)
                        np.save(combined_path, combined)

                        segment_index += 1
                    
                    total_files += 1
                    total_segments += segment_index
                    print(f"    -> {segment_index} segments saved from {file_path}")

                except Exception as e:
                    logging.error(f"Error processing {file_path}: {e}")
                    total_errors += 1

    end_time = time.time()
    elapsed_time = end_time - start_time

    print("\n==================== PREPROCESSING COMPLETED ====================\n")

    print(f"Total files processed: {total_files}")
    print(f"Total segments saved: {total_segments}")
    print(f"Total errors: {total_errors}")
    print(f"Total time taken: {elapsed_time:.2f} seconds")

    print("\n===============================================================")