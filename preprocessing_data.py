import os
import numpy as np
import torchaudio

DB_PATH = "DB/"
SAVE_PATH = "preprocessed_data/"
NB_TIME = 16000 * 1  # 3 seconds (modify as needed)
SEGMENT_STRIDE = NB_TIME // 2  # 50% overlap

os.makedirs(SAVE_PATH, exist_ok=True)

for subfolder in ["synthetic voice", "real voice"]:
    folder_path = os.path.join(DB_PATH, subfolder)
    save_folder = os.path.join(SAVE_PATH, subfolder)
    os.makedirs(save_folder, exist_ok=True)

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".wav"):
            file_path = os.path.join(folder_path, file_name)
            waveform, _ = torchaudio.load(file_path)

            # Convert to numpy array
            audio_np = waveform.numpy()

            # Process in fixed-length overlapping segments
            num_segments = (audio_np.shape[1] - NB_TIME) // SEGMENT_STRIDE + 1
            segment_count = 0

            for i in range(num_segments):
                start_idx = i * SEGMENT_STRIDE
                end_idx = start_idx + NB_TIME

                if end_idx <= audio_np.shape[1]:  # Ensure segment is full-length
                    segment = audio_np[:, start_idx:end_idx]

                    # Save as .npy with segment numbering
                    segment_filename = f"{file_name.replace('.wav', '')}_{segment_count}.npy"
                    np.save(os.path.join(save_folder, segment_filename), segment)

                    segment_count += 1

            print(f"Processed {file_name}: {segment_count} segments")
