"""Audio preprocessing utilities for Bio-ARN."""

from __future__ import annotations

from dataclasses import replace

import torch
import torch.nn.functional as F

from bioarn.config import AudioConfig


class AudioPreprocessor:
    """Convert raw waveforms into fixed-size mel spectrogram features."""

    def __init__(self, config: AudioConfig):
        self.config = replace(config)
        self.n_mels: int = int(self.config.n_mels)
        self.n_fft: int = int(self.config.n_fft)
        self.hop_length: int = int(self.config.hop_length)
        self.sample_rate: int = int(self.config.sample_rate)
        self._mel_cache: dict[tuple[str, str], torch.Tensor] = {}

    @property
    def max_samples(self) -> int:
        return int(self.config.max_samples)

    @property
    def max_frames(self) -> int:
        return int(self.config.max_frames)

    @staticmethod
    def _hz_to_mel(frequencies: torch.Tensor) -> torch.Tensor:
        return 2595.0 * torch.log10(1.0 + (frequencies / 700.0))

    @staticmethod
    def _mel_to_hz(mels: torch.Tensor) -> torch.Tensor:
        return 700.0 * (torch.pow(10.0, mels / 2595.0) - 1.0)

    def _cache_key(self, device: torch.device, dtype: torch.dtype) -> tuple[str, str]:
        return (str(device), str(dtype))

    def _mel_filterbank(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = self._cache_key(device, dtype)
        cached = self._mel_cache.get(key)
        if cached is not None:
            return cached

        n_freqs = (self.n_fft // 2) + 1
        hz_points = self._mel_to_hz(
            torch.linspace(
                self._hz_to_mel(torch.tensor(0.0)),
                self._hz_to_mel(torch.tensor(self.sample_rate / 2.0)),
                self.n_mels + 2,
                dtype=torch.float32,
                device=device,
            )
        )
        fft_bins = hz_points * (float(self.n_fft) / float(self.sample_rate))
        frequencies = torch.arange(n_freqs, dtype=torch.float32, device=device)
        filterbank = torch.zeros(self.n_mels, n_freqs, dtype=torch.float32, device=device)

        for index in range(self.n_mels):
            left = fft_bins[index]
            center = fft_bins[index + 1]
            right = fft_bins[index + 2]
            up_slope = (frequencies - left) / max(float(center - left), 1e-6)
            down_slope = (right - frequencies) / max(float(right - center), 1e-6)
            filterbank[index] = torch.clamp(torch.minimum(up_slope, down_slope), min=0.0)

        filterbank = filterbank / filterbank.sum(dim=1, keepdim=True).clamp_min(1e-6)
        self._mel_cache[key] = filterbank.to(dtype=dtype)
        return self._mel_cache[key]

    def _prepare_waveform(self, waveform: torch.Tensor) -> torch.Tensor:
        prepared = waveform.to(torch.float32)
        if prepared.dim() == 2 and prepared.shape[0] == 1:
            prepared = prepared.squeeze(0)
        elif prepared.dim() == 2:
            prepared = prepared.mean(dim=0)
        elif prepared.dim() != 1:
            raise ValueError("waveform must have shape (samples,) or (channels, samples).")

        if prepared.numel() > self.max_samples:
            prepared = prepared[: self.max_samples]
        elif prepared.numel() < self.max_samples:
            prepared = F.pad(prepared, (0, self.max_samples - prepared.numel()))
        return prepared

    def waveform_to_mel(self, waveform: torch.Tensor) -> torch.Tensor:
        """Convert ``(samples,)`` into a log-mel spectrogram ``(n_mels, frames)``."""

        prepared = self._prepare_waveform(waveform)
        window = torch.hann_window(self.n_fft, device=prepared.device, dtype=prepared.dtype)
        stft = torch.stft(
            prepared,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=window,
            return_complex=True,
            center=True,
        )
        power = stft.abs().pow(2.0)
        mel_filterbank = self._mel_filterbank(device=power.device, dtype=power.dtype)
        mel = mel_filterbank @ power
        mel = torch.log1p(mel)
        return mel.to(torch.float32)

    def to_flat_input(self, mel: torch.Tensor) -> torch.Tensor:
        """Pad or trim mel spectrograms to a fixed frame count, then flatten."""

        prepared = mel.to(torch.float32)
        squeeze = False
        if prepared.dim() == 2:
            prepared = prepared.unsqueeze(0)
            squeeze = True
        if prepared.dim() != 3 or prepared.shape[1] != self.n_mels:
            raise ValueError("mel must have shape (n_mels, frames) or (batch, n_mels, frames).")

        frame_count = prepared.shape[-1]
        if frame_count > self.max_frames:
            prepared = prepared[..., : self.max_frames]
        elif frame_count < self.max_frames:
            prepared = F.pad(prepared, (0, self.max_frames - frame_count))

        flattened = prepared.reshape(prepared.shape[0], -1)
        return flattened.squeeze(0) if squeeze else flattened


__all__ = ["AudioPreprocessor"]
