import os
import sys
import time
import torchaudio
import numpy as np
from pydub import AudioSegment
from utils.Logger import Logger
from datetime import datetime
from classes.models.PathologicalFeature.PathologicalFeatureExtractor import PathologicalFeatureExtractor

DB_PATH = "unseen_dataset/"
TEMP_SEGMENT_PATH = "temp_segments/"
PREPROC_PATH = "test_preprocessed_data/"
SECONDS = 4                                         # Length of each segment in seconds  
NB_TIME = 16000 * SECONDS
SEGMENT_LENGTH_MS = SECONDS * 1000                  # 4 seconds in milliseconds
SEGMENT_STRIDE_AUDIO = SEGMENT_LENGTH_MS // 4       # 25% overlap (1 second overlap for 4 seconds)

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
    total_errors = 0
    total_segments_overall = 0

    extractor = PathologicalFeatureExtractor()

    preprocessing_start_time = time.time()
    print("==================== PREPROCESSING STARTED ====================\n")
    for label in ["Spoof", "Bonafide"]:
        label_path = os.path.join(DB_PATH, label)
        for root, dirs, files in os.walk(label_path):
            for file_name in files:
                if not file_name.endswith(".wav"):
                    continue

                file_path = os.path.join(root, file_name)
                try:
                    print(f">> Processing file: {file_path}")
                    # Compute relative path
                    rel_path = os.path.relpath(root, DB_PATH)

                    # Define base save dirs per category and ensure they exist
                    waveform_dir = os.path.join(PREPROC_PATH, "waveform", rel_path)
                    patho_dir = os.path.join(PREPROC_PATH, "patho", rel_path)
                    combined_dir = os.path.join(PREPROC_PATH, "combined", rel_path)
                    os.makedirs(waveform_dir, exist_ok=True)
                    os.makedirs(patho_dir, exist_ok=True)
                    os.makedirs(combined_dir, exist_ok=True)

                    # for raw waveform
                    waveform, sr = torchaudio.load(file_path)
                    if waveform.shape[0] > 1:
                        waveform = waveform.mean(dim=0, keepdim=True)
                    waveform = waveform.squeeze(0)
                    T = waveform.shape[0]

                    # chunk the audio file into segments for extracting pathological features
                    audio = AudioSegment.from_wav(file_path)
                    duration_ms = len(audio)

                    # Folder to save chunks of this file
                    base_name = os.path.splitext(file_name)[0]
                    out_dir = os.path.join(TEMP_SEGMENT_PATH, label, base_name)
                    os.makedirs(out_dir, exist_ok=True)

                    chunk_paths = []
                    num_chunks = 0
                    for i in range(0, duration_ms, SEGMENT_STRIDE_AUDIO):
                        # Only save if the available duration left is >= half the segment length
                        if duration_ms - i < SEGMENT_LENGTH_MS // 2:
                            break
                        chunk = audio[i:i + SEGMENT_LENGTH_MS]
                        chunk_filename = f"{base_name}_segment_{i // SEGMENT_LENGTH_MS}.wav"
                        chunk_path = os.path.join(out_dir, chunk_filename)
                        chunk.export(chunk_path, format="wav")
                        chunk_paths.append(chunk_path)
                        num_chunks += 1

                    print(f"    >> {file_name}: {num_chunks} chunks generated \n        >> (duration: {duration_ms} ms, segment length: {SEGMENT_LENGTH_MS} ms, stride: {SEGMENT_STRIDE_AUDIO} ms)\n")

                    total_segments = 0
                    for segment_idx, chunk_path in enumerate(chunk_paths):
                        # Extract pathological features
                        patho_feat = extractor(chunk_path).float()

                        start_time = segment_idx * SEGMENT_STRIDE_AUDIO

                        # Safely slice waveform, pad if needed
                        end_time = start_time + NB_TIME
                        if start_time < waveform.shape[0]:
                            if end_time > waveform.shape[0]:
                                # Pad with the edge value (same padding) if segment is short
                                pad_length = end_time - waveform.shape[0]
                                waveform_feat = np.pad(waveform[start_time:], (0, pad_length), mode="edge")
                            else:
                                waveform_feat = waveform[start_time:end_time]

                        patho_feat_np = patho_feat.numpy()
                        waveform_feat_np = waveform_feat
                        combined = np.concatenate([waveform_feat_np, patho_feat_np], axis=0)

                        # Base file naming
                        base_name = os.path.splitext(file_name)[0]
                        save_base = f"{base_name}_{segment_idx}"
                        
                        # Save paths (in subfolders)
                        waveform_path = os.path.join(waveform_dir, f"{save_base}.npy")
                        patho_path = os.path.join(patho_dir, f"{save_base}.npy")
                        combined_path = os.path.join(combined_dir, f"{save_base}.npy")

                        # Save the files
                        np.save(waveform_path, waveform_feat_np)
                        np.save(patho_path, patho_feat_np)
                        np.save(combined_path, combined)
                        
                        total_segments += 1

                    total_files += 1
                    total_segments_overall += total_segments

                    # Remove temporary chunk files and directory after processing
                    for chunk_path in chunk_paths:
                        if os.path.exists(chunk_path):
                            os.remove(chunk_path)
                    if os.path.exists(out_dir) and len(os.listdir(out_dir)) == 0:
                        os.rmdir(out_dir)

                    # print(f"    -> {total_segments} segments saved from {file_path}")

                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
                    total_errors += 1

    preprocessing_end_time = time.time()
    elapsed_preprocessing_time = preprocessing_end_time - preprocessing_start_time

    print("\n==================== PREPROCESSING COMPLETED ====================\n")

    print(f"Total files processed: {total_files}")
    print(f"Total segments saved: {total_segments_overall}")
    print(f"Total errors: {total_errors}")
    print(f"Total time taken: {elapsed_preprocessing_time:.2f} seconds")

    print("\n===============================================================")