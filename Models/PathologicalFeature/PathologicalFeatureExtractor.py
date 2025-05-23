import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import parselmouth
import librosa
from scipy.fftpack import fft, ifft
from scipy.signal import hilbert, resample, lfilter
from scipy.linalg import solve_toeplitz

class VoiceFeatureExtractor:
    def __init__(self, sr=16000):
        self.sr = sr

    def extract(self, segment):
        print("Extracting features...")
        snd = parselmouth.Sound(segment, self.sr)
        print("Extracting snd...")
        pp = parselmouth.praat.call(snd, "To PointProcess (periodic, cc)", 0.1, 500)
        print("Extracting pp...")

        # Extract jitter and shimmer features
        jitter_local = parselmouth.praat.call([snd, pp], "Get jitter (local)", 0.0001, 0.02, 1.3)
        print(f"Extracting jitter_local: {jitter_local}")

        jitter_ppq3 = parselmouth.praat.call([snd, pp], "Get jitter (ppq5)", 0.0001, 0.02, 1.3)
        print(f"Extracting jitter_ppq3: {jitter_ppq3}")

        jitter_ppq5 = parselmouth.praat.call([snd, pp], "Get jitter (ppq5)", 0.0001, 0.02, 1.3)
        print(f"Extracting jitter_ppq5: {jitter_ppq5}")

        shimmer_local = parselmouth.praat.call([snd, pp], "Get shimmer (local)", 0.0001, 0.02, 1.3, 1.6)
        print(f"Extracting shimmer_local: {shimmer_local}")

        shimmer_apq3 = parselmouth.praat.call([snd, pp], "Get shimmer (apq3)", 0.0001, 0.02, 1.3, 1.6)
        print(f"Extracting shimmer_apq3: {shimmer_apq3}")

        shimmer_apq5 = parselmouth.praat.call([snd, pp], "Get shimmer (apq5)", 0.0001, 0.02, 1.3, 1.6)
        print(f"Extracting shimmer_apq5: {shimmer_apq5}")

        shimmer_apq11 = parselmouth.praat.call([snd, pp], "Get shimmer (apq11)", 0.0001, 0.02, 1.3, 1.6)
        print(f"Extracting shimmer_apq11: {shimmer_apq11}")

        segment_np = segment if isinstance(segment, np.ndarray) else segment.values
        chnr = self.compute_chnr(segment_np)
        print("Extracting chnr...")
        nne  = self.compute_nne(segment_np)
        print("Extracting nne...")
        gne  = self.compute_gne(segment_np)
        print("Extracting gne...")

        return np.array([
            jitter_local, jitter_ppq3, jitter_ppq5,
            shimmer_local, shimmer_apq3, shimmer_apq5, shimmer_apq11,
            chnr, nne, gne
        ])

    def compute_chnr(self, signal):
        frame_len = int(0.025 * self.sr)
        if len(signal) < frame_len:
            return 0
        frame = signal[:frame_len] * np.hamming(frame_len)

        spectrum = np.abs(fft(frame))
        log_spectrum = np.log(spectrum + 1e-10)
        cepstrum = np.real(ifft(log_spectrum))

        harmonic_energy = np.sum(cepstrum[1:20] ** 2)  # Harmonic part
        total_energy = np.sum(cepstrum ** 2)           # Cepstral total energy

        # Filtering step would ideally reduce harmonic influence
        # Noise is everything minus harmonic component
        noise_energy = total_energy - harmonic_energy

        # Level difference = total_energy - noise_energy = harmonic_energy
        # CHNR = harmonic_energy / noise_energy
        return 10 * np.log10((harmonic_energy + 1e-6) / (noise_energy + 1e-6))

    def compute_nne(self, signal):
        print("NNE...")
        f0, voiced_flag, _ = librosa.pyin(signal, fmin=75, fmax=500, sr=self.sr)
        print("Extracting f0...")
        pitch = np.nanmean(f0)
        print("Extracting pitch...")
        if np.isnan(pitch):
            pitch = 150
        print("Extracting pitch...")
        
        frame_len = int(0.025 * self.sr)
        if len(signal) < frame_len:
            return 0
        frame = signal[:frame_len] * np.hamming(frame_len)
        print("Extracting frame...")
        S = np.abs(fft(frame))[:frame_len // 2]
        print("Extracting S...")
        log_S = np.log(S + 1e-10)
        print("Extracting log_S...")

        total_energy = np.sum(log_S ** 2)
        print("Extracting total_energy...")
        harmonic_bins = np.array([int(pitch * i / (self.sr / frame_len)) for i in range(1, int((self.sr / 2) // pitch))])
        print("Extracting harmonic_bins...")
        noise_bins = np.setdiff1d(np.arange(len(log_S)), harmonic_bins)
        print("Extracting noise_bins...")
        noise_energy = np.sum(log_S[noise_bins] ** 2)
        print("Extracting noise_energy...")

        return noise_energy / (total_energy + 1e-10)

    def compute_gne(self, signal):
        target_sr = 10000
        if len(signal) < int(0.025 * self.sr):
            return 0

        # Downsample to 10 kHz
        resampled = resample(signal, int(len(signal) * target_sr / self.sr))

        # Inverse Filtering using LPC
        frame = resampled[:int(0.025 * target_sr)] * np.hamming(int(0.025 * target_sr))
        order = 16  # LPC order

        # LPC coefficient computation (Autocorrelation method)
        autocorr = np.correlate(frame, frame, mode='full')
        autocorr = autocorr[len(autocorr) // 2:]
        R = autocorr[:order + 1]
        a = solve_toeplitz((R[:-1], R[:-1]), R[1:])  # LPC coeffs (without leading 1)
        a = np.concatenate(([1], -a))  # Full LPC filter

        # Apply inverse filter
        residual = lfilter(a, [1.0], frame)

        # Compute maximum autocorrelation (as a proxy for voicing strength)
        max_corr = np.max(np.correlate(residual, residual, mode='full'))

        # Hilbert envelopes
        envelope = np.abs(hilbert(residual))
        mid = len(envelope) // 2
        env1, env2 = envelope[:mid], envelope[mid:]

        if len(env1) != len(env2):
            return 0

        # Cross-correlation of envelopes
        corr = np.correlate(env1 - np.mean(env1), env2 - np.mean(env2), mode='valid')
        envelope_corr = np.max(corr) / (np.std(env1) * np.std(env2) + 1e-6)

        # Combine into final GNE score (you may optionally normalize by max_corr)
        return envelope_corr

# -------------------------
# Pathology Extraction Modules
# -------------------------

class PathologicalFeatureExtractor(nn.Module):
    def __init__(self, sr=16000, frame_length=0.5, hop_length=0.25):
        super(PathologicalFeatureExtractor, self).__init__()
        self.feature_extractor = VoiceFeatureExtractor(sr=sr)
        self.sr = sr
        self.frame_len = int(frame_length * sr)
        self.hop_len = int(hop_length * sr)

    def compute_deltas(self, features):
        delta = features[:, :, 1:] - features[:, :, :-1]
        delta = F.pad(delta, (1, 0), mode='replicate')
        delta2 = delta[:, :, 1:] - delta[:, :, :-1]
        delta2 = F.pad(delta2, (1, 0), mode='replicate')
        return delta, delta2

    def forward(self, x):
        # x shape: [B, 1, T]
        B, C, T = x.shape
        assert C == 1, "Expected single-channel waveform input"

        feature_list = []

        for b in range(B):
            segment = x[b, 0].cpu().numpy()
            features = []

            # Segment and extract features
            for start in range(0, T - self.frame_len + 1, self.hop_len):
                frame = segment[start:start + self.frame_len]
                feat = self.feature_extractor.extract(frame)
                features.append(feat)

            features = np.stack(features, axis=0)  # [T', 10]
            feature_list.append(torch.tensor(features.T, dtype=torch.float32))  # [10, T']

        out = torch.stack(feature_list, dim=0).to(x.device)  # [B, 10, T']
        delta, delta2 = self.compute_deltas(out)
        return torch.cat([out, delta, delta2], dim=1)  # [B, 30, T']