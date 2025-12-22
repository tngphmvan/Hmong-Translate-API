---
license: cc-by-nc-nd-4.0
---
# Vietnamese Speech-to-Text (ASR) — ZipFormer-30M-RNNT-6000h

## 🔍 Overview
The **Vietnamese Speech-to-Text (ASR)** model is built on the **ZipFormer architecture** — an improved variant of the Conformer — featuring only **30 million parameters** yet achieving **exceptional performance** in both speed and accuracy.  
On CPU, the model can transcribe a **12-second audio clip in just 0.3 seconds**, significantly faster than most traditional ASR systems without requiring a GPU.

---

## 🚀 Online Demo

You can try the model directly here:  
👉 https://huggingface.co/spaces/hynt/k2-automatic-speech-recognition-demo

---

## ⚙️ Model Architecture and Training strategy:
- **Architecture:** ZipFormer  
- **Parameters:** ~30M  
- **Language:** Vietnamese  
- **Loss Function:** RNN-Transducer (RNNT Loss)  
- **Framework:** PyTorch + k2
- **Training strategy**: Carefully preprocess the data, apply an augmentation strategy based on the distribution of out-of-vocabulary (OOV) tokens and refine the transcriptions using Whisper.  
- **Optimized for:** High-speed CPU inference  

---

## 🧠 Training Data
The model was trained on approximately **6000 hours of high-quality Vietnamese speech** collected from various public datasets:

| Dataset |  |  |
|----------|----------|----------|
| VLSP2020 | VLSP2021 | VLSP2023-voting-pseudo-labeled |
| VLSP2023 | FPT | VIET_BUD500 |
| VietSpeech | FLEURS | VietMed_Labeled |
| Sub-GigaSpeech2-Vi | ViVoice | Sub-PhoAudioBook |

---

## 🧪 Evaluation Results

| **Dataset** | **ZipFormer-30M-6000h** | **ChunkFormer-110M-3000h** | **PhoWhisper-Large-1.5B-800h** | **VietASR-ZipFormer-68M-70.000h** |
|--------------|--------------------------|-----------------------------|--------------------------------|---------------------------------|
| **VLSP2020-Test-T1** | **12.29** | 14.09 | 13.75 | 14.45 |
| **VLSP2023-PublicTest** | **10.40** | 16.15 | 16.83 | 14.70 |
| **VLSP2023-PrivateTest** | **11.10** | 17.12 | 17.10 | 15.07 |
| **VLSP2025-PublicTest** | **7.97** | 15.55 | 16.14 | 13.55 |
| **VLSP2025-PrivateTest** | **8.10** | 16.07 | 16.31 | 13.97 |
| **GigaSpeech2-Test** | 7.56 | 10.35 | 10.00 | **6.88** |

> Lower is better (WER %)

---

## 🏆 Achievements
By training this model architecture on 4,000 hours of data, I **won First Place** in the **Vietnamese Language Speech Processing (VLSP)** competition **2025**.
Comprehensive details about **training data**, **optimization strategies**, **architecture improvements**, and **evaluation methodologies** are available in the paper below:

👉 [Read the full paper on Overleaf](https://www.overleaf.com/read/wjntrgchhbgv#48aa25)

---

## ⚡ Inference Speed

| **Device** | **Audio Length** | **Inference Time** |
|-------------|------------------|--------------------|
| CPU (Hugging Face Basic) | 12 seconds | **0.3 s** |
| GPU (RTX 3090) | 12 seconds | **< 0.1 s** |

---

## ⚙️ How to Run This Model

Please refer to the following guides for instructions on how to run and deploy this model:  
- **For Torch JIT Script:** [https://k2-fsa.github.io/sherpa/](https://k2-fsa.github.io/sherpa/)  
- **For ONNX:** [https://k2-fsa.github.io/sherpa/onnx/](https://k2-fsa.github.io/sherpa/onnx/)

## 💬 Summary
The **ZipFormer-30M-RNNT-6000h** model demonstrates that a lightweight architecture can still achieve state-of-the-art accuracy for Vietnamese ASR.  
It is designed for **fast deployment on CPU-based systems**, making it ideal for **real-time speech recognition**, **callbots**, and **embedded speech interfaces**.

---