"""Mel-spectrogram processing utilities for VITS Text-to-Speech.

This module provides functions for converting audio waveforms to mel-spectrograms
and related audio processing operations. It includes optimized PyTorch implementations
for efficient GPU-accelerated audio feature extraction.

Key functions:
    - spectrogram_torch: Compute STFT spectrogram from waveform
    - spec_to_mel_torch: Convert spectrogram to mel-spectrogram
    - mel_spectrogram_torch: Direct waveform to mel-spectrogram conversion
    - dynamic_range_compression_torch: Log compression for spectrograms
    - dynamic_range_decompression_torch: Inverse of log compression

The module uses caching for Hann windows and mel filter banks to avoid
recomputation across multiple calls with the same parameters.
"""

import math
import os
import random
import torch
try:
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
except:
    device = torch.device('cpu')
from torch import nn
import torch.nn.functional as F
import torch.utils.data
import numpy as np
import librosa
import librosa.util as librosa_util
from librosa.util import normalize, pad_center, tiny
from scipy.signal import get_window
from scipy.io.wavfile import read
from librosa.filters import mel as librosa_mel_fn

MAX_WAV_VALUE = 32768.0


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    """
    PARAMS
    ------
    C: compression factor
    """
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    """
    PARAMS
    ------
    C: compression factor used to compress
    """
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    """Apply spectral normalization (log compression) to magnitude spectrogram.

    Args:
        magnitudes (torch.Tensor): Input magnitude spectrogram.

    Returns:
        torch.Tensor: Log-compressed spectrogram.
    """
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    """Inverse spectral normalization (exponential) for magnitude spectrogram.

    Args:
        magnitudes (torch.Tensor): Log-compressed spectrogram.

    Returns:
        torch.Tensor: Linear magnitude spectrogram.
    """
    output = dynamic_range_decompression_torch(magnitudes)
    return output


mel_basis = {}
hann_window = {}


def spectrogram_torch(y, n_fft, sampling_rate, hop_size, win_size, center=False):
    """Compute magnitude spectrogram from waveform using STFT.

    Uses PyTorch's STFT implementation with a Hann window. Windows are cached
    for efficiency across multiple calls with the same parameters.

    Args:
        y (torch.Tensor): Input waveform tensor of shape [batch, time].
        n_fft (int): FFT size.
        sampling_rate (int): Audio sampling rate (not directly used but kept for API consistency).
        hop_size (int): Hop length between STFT frames.
        win_size (int): Window size for STFT.
        center (bool, optional): Whether to center the signal. Defaults to False.

    Returns:
        torch.Tensor: Magnitude spectrogram of shape [batch, n_fft//2+1, frames].
    """
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))

    global hann_window
    dtype_device = str(y.dtype) + '_' + str(y.device)
    wnsize_dtype_device = str(win_size) + '_' + dtype_device
    if wnsize_dtype_device not in hann_window:
        hann_window[wnsize_dtype_device] = torch.hann_window(
            win_size).to(dtype=y.dtype, device=y.device)

    y = torch.nn.functional.pad(y.unsqueeze(
        1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)

    spec = torch.stft(y, n_fft, return_complex=False, hop_length=hop_size, win_length=win_size, window=hann_window[wnsize_dtype_device],
                      center=center, pad_mode='reflect', normalized=False, onesided=True)

    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)
    return spec


def spec_to_mel_torch(spec, n_fft, num_mels, sampling_rate, fmin, fmax):
    """Convert linear spectrogram to mel-spectrogram.

    Applies mel filter bank to linear spectrogram and performs spectral normalization.
    Mel filter banks are cached for efficiency.

    Args:
        spec (torch.Tensor): Linear spectrogram of shape [batch, n_fft//2+1, frames].
        n_fft (int): FFT size used to compute the spectrogram.
        num_mels (int): Number of mel frequency bins.
        sampling_rate (int): Audio sampling rate.
        fmin (float): Minimum frequency for mel filter bank.
        fmax (float): Maximum frequency for mel filter bank.

    Returns:
        torch.Tensor: Mel-spectrogram of shape [batch, num_mels, frames].
    """
    global mel_basis
    dtype_device = str(spec.dtype) + '_' + str(spec.device)
    fmax_dtype_device = str(fmax) + '_' + dtype_device
    if fmax_dtype_device not in mel_basis:
        mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
        mel_basis[fmax_dtype_device] = torch.from_numpy(
            mel).to(dtype=spec.dtype, device=spec.device)
    spec = torch.matmul(mel_basis[fmax_dtype_device], spec)
    spec = spectral_normalize_torch(spec)
    return spec


def mel_spectrogram_torch(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    """Compute mel-spectrogram directly from waveform.

    Combines STFT computation and mel filter bank application into a single
    optimized function. Caches Hann windows and mel filter banks for efficiency.

    Args:
        y (torch.Tensor): Input waveform tensor of shape [batch, time].
        n_fft (int): FFT size.
        num_mels (int): Number of mel frequency bins.
        sampling_rate (int): Audio sampling rate.
        hop_size (int): Hop length between STFT frames.
        win_size (int): Window size for STFT.
        fmin (float): Minimum frequency for mel filter bank.
        fmax (float): Maximum frequency for mel filter bank.
        center (bool, optional): Whether to center the signal. Defaults to False.

    Returns:
        torch.Tensor: Mel-spectrogram of shape [batch, num_mels, frames].
    """
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))

    global mel_basis, hann_window
    dtype_device = str(y.dtype) + '_' + str(y.device)
    fmax_dtype_device = str(fmax) + '_' + dtype_device
    wnsize_dtype_device = str(win_size) + '_' + dtype_device
    if fmax_dtype_device not in mel_basis:
        mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
        mel_basis[fmax_dtype_device] = torch.from_numpy(
            mel).to(dtype=y.dtype, device=y.device)
    if wnsize_dtype_device not in hann_window:
        hann_window[wnsize_dtype_device] = torch.hann_window(
            win_size).to(dtype=y.dtype, device=y.device)

    y = torch.nn.functional.pad(y.unsqueeze(
        1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)

    spec = torch.stft(y, n_fft, return_complex=False, hop_length=hop_size, win_length=win_size, window=hann_window[wnsize_dtype_device],
                      center=center, pad_mode='reflect', normalized=False, onesided=True)

    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)

    spec = torch.matmul(mel_basis[fmax_dtype_device], spec)
    spec = spectral_normalize_torch(spec)

    return spec
