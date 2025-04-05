import os
import numpy as np
import torchaudio

DB_PATH = "DB/"
SAVE_PATH = "preprocessed_data/"
NB_TIME = 16000 * 1  # 1 second (modify as needed)
SEGMENT_STRIDE = NB_TIME // 2  # 50% overlap

# Make sure the correct backend is used for decoding mp3
torchaudio.set_audio_backend("sox_io")  # or "ffmpeg" if installed and preferred

os.makedirs(SAVE_PATH, exist_ok=True)

for subfolder in ["synthetic voice", "real voice"]:
    folder_path = os.path.join(DB_PATH, subfolder)
    save_folder = os.path.join(SAVE_PATH, subfolder)
    os.makedirs(save_folder, exist_ok=True)

    for file_name in os.listdir(folder_path):
        if file_name.endswith((".wav", ".mp3")):  # support both formats
            file_path = os.path.join(folder_path, file_name)
            waveform, sample_rate = torchaudio.load(file_path)

            # Resample if not 16kHz
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
                waveform = resampler(waveform)
                sample_rate = 16000

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
                    base_name = os.path.splitext(file_name)[0]
                    segment_filename = f"{base_name}_{segment_count}.npy"
                    np.save(os.path.join(save_folder, segment_filename), segment)

                    segment_count += 1

            print(f"Processed {file_name}: {segment_count} segments")
