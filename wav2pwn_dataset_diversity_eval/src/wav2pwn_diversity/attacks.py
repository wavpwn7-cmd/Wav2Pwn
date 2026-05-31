from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch

from .metrics import normalize_text


@dataclass
class AttackResult:
    adversarial_waveform: torch.Tensor
    num_iterations: int
    elapsed_seconds: float
    final_loss: float


def _build_labels(processor, text: str, device: torch.device) -> torch.Tensor:
    encoded = processor.tokenizer(normalize_text(text), return_tensors="pt")
    return encoded.input_ids.to(device)


def _ctc_loss_for_waveform(model, processor, waveform: torch.Tensor, labels: torch.Tensor, device: torch.device) -> torch.Tensor:
    input_values = waveform.unsqueeze(0)
    attention_mask = torch.ones_like(input_values, dtype=torch.long, device=device)
    logits = model(input_values=input_values, attention_mask=attention_mask).logits
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1).transpose(0, 1)
    input_lengths = torch.full(
        size=(log_probs.size(1),),
        fill_value=log_probs.size(0),
        dtype=torch.long,
        device=device,
    )
    target_lengths = torch.full(
        size=(labels.size(0),),
        fill_value=labels.size(1),
        dtype=torch.long,
        device=device,
    )
    return torch.nn.functional.ctc_loss(
        log_probs,
        labels,
        input_lengths,
        target_lengths,
        blank=processor.tokenizer.pad_token_id,
        zero_infinity=True,
    )


def optimize_attack(
    model,
    processor,
    waveform: torch.Tensor,
    reference_text: str,
    epsilon: float,
    alpha: float,
    num_iterations: int,
    targeted: bool,
    target_text: str | None,
    device: torch.device,
    method: str = "adam",
    l2_weight: float = 0.0,
    random_start: bool = False,
) -> AttackResult:
    start_time = time.perf_counter()
    model.eval()

    clean = waveform.detach().to(device)
    objective_text = target_text if targeted else reference_text
    if objective_text is None:
        raise ValueError("Target or reference transcription must be provided.")
    labels = _build_labels(processor, objective_text, device)
    last_loss = 0.0

    method = method.lower()
    if method == "pgd":
        adv = clean.clone().detach()
        if random_start:
            adv = torch.clamp(clean + torch.empty_like(clean).uniform_(-epsilon, epsilon), -1.0, 1.0).detach()
        adv.requires_grad_(True)
        for _ in range(num_iterations):
            model.zero_grad(set_to_none=True)
            loss = _ctc_loss_for_waveform(model, processor, adv, labels, device)
            optimize_loss = loss if targeted else -loss
            optimize_loss.backward()
            gradient = adv.grad.sign()
            adv = adv.detach() - alpha * gradient if targeted else adv.detach() + alpha * gradient
            perturbation = torch.clamp(adv - clean, min=-epsilon, max=epsilon)
            adv = torch.clamp(clean + perturbation, min=-1.0, max=1.0).detach()
            adv.requires_grad_(True)
            last_loss = float(loss.detach().item())
    else:
        perturbation = torch.zeros_like(clean, device=device, requires_grad=True)
        if random_start:
            with torch.no_grad():
                perturbation.uniform_(-epsilon, epsilon)
        optimizer = torch.optim.Adam([perturbation], lr=alpha)
        for _ in range(num_iterations):
            adv = torch.clamp(clean + perturbation, min=-1.0, max=1.0)
            model.zero_grad(set_to_none=True)
            optimizer.zero_grad(set_to_none=True)
            loss = _ctc_loss_for_waveform(model, processor, adv, labels, device)
            reg = l2_weight * torch.norm(perturbation, p=2)
            optimize_loss = loss + reg if targeted else -loss + reg
            optimize_loss.backward()
            optimizer.step()
            with torch.no_grad():
                perturbation.clamp_(-epsilon, epsilon)
            last_loss = float(loss.detach().item())
        adv = torch.clamp(clean + perturbation.detach(), min=-1.0, max=1.0).detach()

    return AttackResult(
        adversarial_waveform=adv.detach().cpu(),
        num_iterations=num_iterations,
        elapsed_seconds=time.perf_counter() - start_time,
        final_loss=last_loss,
    )


def save_waveform(path: str, waveform: torch.Tensor, sample_rate: int = 16000) -> None:
    import soundfile as sf

    audio = waveform.detach().cpu().numpy().astype(np.float32)
    sf.write(path, audio, sample_rate)
