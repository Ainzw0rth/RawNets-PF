# Synthetic Speech Detection using RawNet with Pathological Features  

This repository contains the implementation and experiments from the research paper:  

**Synthetic Speech Detection using RawNet with Pathological Features**  
*Louis Caesa Kesuma, Dessi Puji Lestari, Candy Olivia Mawalim*  
Institut Teknologi Bandung, Indonesia | Japan Advanced Institute of Science and Technology  

---

## 📖 Overview  
With the rise of highly realistic **synthetic speech**, protecting **Automatic Speaker Verification (ASV)** systems has become increasingly important. This project explores how **RawNet models (RawNet1, RawNet2, RawNet3)** can be enhanced with **pathological voice features**—such as jitter, shimmer, and glottal-to-noise excitation ratio (GNE)—to improve **deepfake voice detection**, especially in **Indonesian speech**, a low-resource language in this domain.  

Key contributions:  
- Introduced an **Indonesian speech dataset** with both genuine and synthetic voices.  
- Integrated **pathological features** at both the **feature-level (GF)** and **architectural-level (GA)**.  
- Demonstrated that **RawNet2-GA** achieves the best robustness and accuracy across seen, unseen-in-dataset (UID), and unseen-not-in-dataset (UnID) scenarios.  

The full paper is available in the [`docs/`](./docs) folder.  

---

## 📂 Dataset  
The dataset consists of **39,205 audio samples** (~3.9s average length) sourced from:  
- **Real speech**: [Mozilla Common Voice](https://commonvoice.mozilla.org) (5,000 samples), [Prosa.ai](https://prosa.ai) (2,000 samples)  
- **Synthetic speech**: Generated using multiple TTS and VC systems:  
  - Google TTS  
  - Facebook MMS  
  - VITS  
  - ElevenLabs  
  - DupDub  
  - FreeVC (combined with multiple TTS systems)  

Dataset scenarios:  
1. **Seen** – Training & testing share TTS systems/transcripts.  
2. **UID** – Testing includes unseen TTS systems but same transcripts.  
3. **UnID** – Testing includes unseen TTS systems & transcripts.  

---

## ⚙️ Methods  
We tested **RawNet1, RawNet2, and RawNet3**, with the following setups:  

- **Baseline**: Raw waveform input only.  
- **GF (Feature-level fusion)**: Concatenated raw waveform + pathological features.  
- **GA (Architectural-level fusion)**: Process raw features first, then integrate pathological features in the FC layer.  

Pathological features used:  
- Jitter (local, PPQ3, PPQ5)  
- Shimmer (local, APQ3, APQ5, APQ11)  
- Glottal-to-Noise Excitation Ratio (GNE)  

---

## 📊 Results  
| Model | Accuracy | F1-Score | Best Variant |  
|-------|-----------|----------|--------------|  
| RawNet1 | 87.10% | 89.18% | RawNet1-GA (93.37% F1) |  
| RawNet2 | 95.73% | 95.97% | **RawNet2-GA (97.05% F1)** |  
| RawNet3 | 90.77% | 91.64% | RawNet3-GA (94.31% F1) |  

✅ **RawNet2-GA** consistently outperformed other variants and proved most robust across unseen scenarios.  

---

## 🚀 Installation & Usage  

### 1. Clone the repository  
```bash
git clone https://github.com/Ainzw0rth/RawNets-PF.git
cd RawNets-PF
```

### 2. Create and activate a virtual environment (Python 3.9)  
```bash
py -3.9 -m venv rawnet-pf

source rawnet-pf/Scripts/activate
```

### 3. Install dependencies  
```bash
python -m pip install --upgrade pip setuptools wheel

pip install -r requirements.txt
```

### 4. Train a model  
```bash
python train.py --model rawnet2 --integration GA --epochs 50
```

### 5. Evaluate  
```bash
python evaluate.py --model rawnet2 --integration GA --scenario UID
```

---

## 📌 Citation  
If you use this work, please cite:  

```
@inproceedings{kesuma2024synthetic,
  title={Synthetic Speech Detection using RawNet with Pathological Features},
  author={Kesuma, Louis Caesa and Lestari, Dessi Puji and Mawalim, Candy Olivia},
  booktitle={O-COCOSDA 2024},
  year={2024}
}
```

---

## 👥 Authors  
- **Louis Caesa Kesuma** – Institut Teknologi Bandung  
- **Dessi Puji Lestari** – Institut Teknologi Bandung  
- **Candy Olivia Mawalim** – Japan Advanced Institute of Science and Technology  
