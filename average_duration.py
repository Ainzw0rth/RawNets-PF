import os
import sys
import soundfile as sf
from datetime import datetime
from utils.Logger import Logger

def get_wav_duration(file_path):
    with sf.SoundFile(file_path) as f:
        return len(f) / f.samplerate

def print_wav_durations_and_average(root_dir):
    durations = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith('.wav'):
                file_path = os.path.join(root, file)
                try:
                    duration = get_wav_duration(file_path)
                    durations.append(duration)

                except Exception as e:
                    print(f"Skipping {file_path}: {e}")
    if durations:
        avg_duration = sum(durations) / len(durations)
        print(f"\nAverage duration for {root_dir}: \n       {avg_duration:.2f} seconds \n       over {len(durations)} files.")
    else:
        print("No .wav files found.")

if __name__ == "__main__":
    os.makedirs("logs/utils/", exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"logs/utils/average_duration_{timestamp}.txt"
    log_file = open(log_filename, "w")
    sys.stdout = Logger(sys.stdout, log_file)
    sys.stderr = Logger(sys.stderr, log_file)


    spoof_dirs = [
        "dataset/Spoof/Converted/FacebookMMS",
        "dataset/Spoof/Converted/GoogleTTS",
        "dataset/Spoof/Converted/VITS",
        "dataset/Spoof/TTS/FacebookMMS",
        "dataset/Spoof/TTS/GoogleTTS",
        "dataset/Spoof/TTS/VITS"
    ]
    
    bonafide_dirs = [
        "dataset/Bonafide/CommonVoice",
        "dataset/Bonafide/Prosa"
    ]

    unseen_dirs = [
        "unseen_dataset/Spoof/Converted/DupDub",
        "unseen_dataset/Spoof/Converted/ElevenMultilingualV2",
        "unseen_dataset/Spoof/TTS/DupDub",
        "unseen_dataset/Spoof/TTS/ElevenMultilingualV2",
        "unseen_dataset/Spoof/TTS/DupDub-NotInDataset",
    ]

    for dir in spoof_dirs + bonafide_dirs + unseen_dirs:
        print_wav_durations_and_average(dir)

    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    log_file.close()