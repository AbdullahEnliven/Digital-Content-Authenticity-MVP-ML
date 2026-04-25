from pathlib import Path

import librosa
import numpy as np

from app.ml.voice_auth.config import VoiceAuthSettings


def preprocess_audio_file(path: Path, settings: VoiceAuthSettings) -> np.ndarray:
    """Decode and standardize an audio file for both training and inference.

    The returned waveform is always:
    - mono
    - resampled to settings.sample_rate, 16 kHz by default
    - peak-normalized
    - center-trimmed or zero-padded to settings.clip_duration_seconds, 4 seconds by default
    """
    waveform = decode_audio_file(path, settings.sample_rate)
    waveform = sanitize_waveform(waveform)
    waveform = peak_normalize(waveform)
    return trim_or_pad(waveform, settings.target_num_samples)


def load_and_standardize_audio(path: Path, settings: VoiceAuthSettings) -> np.ndarray:
    """Backward-compatible alias for the shared preprocessing pipeline."""
    return preprocess_audio_file(path, settings)


def decode_audio_file(path: Path, sample_rate: int) -> np.ndarray:
    """Decode FLAC, MP3, WAV, and other common audio files as mono float32."""
    try:
        waveform, _ = librosa.load(path, sr=sample_rate, mono=True)
    except Exception as exc:
        raise ValueError(f"Could not decode audio file '{path.name}': {exc}") from exc

    return np.asarray(waveform, dtype=np.float32)


def sanitize_waveform(waveform: np.ndarray) -> np.ndarray:
    """Remove invalid values and reject empty audio."""
    waveform = np.asarray(waveform, dtype=np.float32).reshape(-1)
    waveform = np.nan_to_num(waveform, nan=0.0, posinf=0.0, neginf=0.0)

    if waveform.size == 0:
        raise ValueError("Audio file contains no readable samples")

    return waveform


def peak_normalize(waveform: np.ndarray) -> np.ndarray:
    """Normalize waveform peak amplitude to the [-1.0, 1.0] range when possible."""
    peak = float(np.max(np.abs(waveform)))
    if peak <= 1e-8:
        return waveform.astype(np.float32, copy=False)
    return (waveform / peak).astype(np.float32, copy=False)


def trim_or_pad(waveform: np.ndarray, target_num_samples: int) -> np.ndarray:
    """Center-trim or zero-pad a waveform to exactly target_num_samples."""
    if waveform.shape[0] == target_num_samples:
        return waveform.astype(np.float32, copy=False)

    if waveform.shape[0] > target_num_samples:
        start = (waveform.shape[0] - target_num_samples) // 2
        return waveform[start : start + target_num_samples].astype(np.float32, copy=False)

    total_padding = target_num_samples - waveform.shape[0]
    left = total_padding // 2
    right = total_padding - left
    return np.pad(waveform, (left, right), mode="constant").astype(np.float32, copy=False)
