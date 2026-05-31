from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
import wave

import numpy as np
import torch
import torchaudio
from transformers import AutoModelForCTC, AutoProcessor


@dataclass
class ASRModelSpec:
    id: str
    hf_name: str
    family: str


@dataclass
class ASRBundle:
    spec: ASRModelSpec
    processor: AutoProcessor
    model: AutoModelForCTC
    device: torch.device


def load_model_registry(path: str | Path) -> list[ASRModelSpec]:
    from .config import load_yaml

    payload = load_yaml(path)
    return [ASRModelSpec(**entry) for entry in payload["models"]]


def load_asr_bundle(spec: ASRModelSpec, device: str | None = None) -> ASRBundle:
    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    processor = AutoProcessor.from_pretrained(spec.hf_name)
    model = AutoModelForCTC.from_pretrained(spec.hf_name).to(resolved_device)
    model.eval()
    return ASRBundle(spec=spec, processor=processor, model=model, device=resolved_device)


def _load_wav_with_wave(path: str) -> tuple[torch.Tensor, int]:
    with wave.open(path, "rb") as wav_handle:
        sample_rate = wav_handle.getframerate()
        num_channels = wav_handle.getnchannels()
        sample_width = wav_handle.getsampwidth()
        num_frames = wav_handle.getnframes()
        raw_audio = wav_handle.readframes(num_frames)

    if sample_width == 2:
        dtype = np.int16
        scale = 32768.0
    elif sample_width == 4:
        dtype = np.int32
        scale = 2147483648.0
    else:
        raise RuntimeError(f"Unsupported WAV sample width {sample_width} for file {path}")

    audio = np.frombuffer(raw_audio, dtype=dtype).astype(np.float32) / scale
    audio = audio.reshape(-1, num_channels).T
    return torch.from_numpy(audio), sample_rate


def load_audio(path: str, target_sr: int = 16000) -> tuple[torch.Tensor, int]:
    suffix = Path(path).suffix.lower()
    waveform = None
    sample_rate = None
    load_errors: list[str] = []

    if suffix == ".wav":
        for attempt in range(3):
            try:
                waveform, sample_rate = _load_wav_with_wave(path)
                break
            except Exception as exc:
                load_errors.append(f"wave_attempt_{attempt + 1}: {exc}")
                time.sleep(0.2)

    if waveform is None or sample_rate is None:
        try:
            import soundfile as sf

            waveform_np, sample_rate = sf.read(path, always_2d=True)
            waveform = torch.from_numpy(waveform_np.T).float()
        except Exception as sf_exc:
            load_errors.append(f"soundfile: {sf_exc}")
            waveform, sample_rate = torchaudio.load(path)

    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != target_sr:
        waveform = torchaudio.functional.resample(waveform, sample_rate, target_sr)
        sample_rate = target_sr
    return waveform.squeeze(0), sample_rate


@torch.inference_mode()
def transcribe_waveform(bundle: ASRBundle, waveform: torch.Tensor, sample_rate: int = 16000) -> str:
    inputs = bundle.processor(
        waveform.cpu().numpy(),
        sampling_rate=sample_rate,
        return_tensors="pt",
        padding=True,
    )
    input_values = inputs.input_values.to(bundle.device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(bundle.device)
    logits = bundle.model(input_values=input_values, attention_mask=attention_mask).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    return bundle.processor.batch_decode(predicted_ids)[0].strip()
