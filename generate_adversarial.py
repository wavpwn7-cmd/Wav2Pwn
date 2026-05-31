# import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import librosa
import soundfile as sf  # For saving audio files
from tqdm import tqdm
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
import torch    

def split_target_text(text):
    # Split text into characters and use | for spaces
    return " ".join(list(text.replace(" ", "|").upper()))

# Configuration
#INPUT_AUDIO_PATH = "/home/mod/Documents/Audio-attack_statistic/fackbook_wav2_target_attack/LibriSpeech/dev-clean/251/118436/251-118436-0000.flac"
INPUT_AUDIO_PATH = "/home/shuhaoz/Wav2PWN-Project-main/LJ-Speech-10-sample/LJ001-0001.wav"
OUTPUT_AUDIO_PATH = "/home/shuhaoz/Wav2PWN-Project-main/Result/perturbed_audio_001.wav"
TARGET_TEXT = "HELLO WORLD"
EPSILON = 0.05  # Initial perturbation strength
lr = 0.001
total_steps = 1000
rw = 0.1  # Regularization weight
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load model and processor
name = "Facebook/wav2vec2-large-960h"
model = Wav2Vec2ForCTC.from_pretrained(name)
processor = Wav2Vec2Processor.from_pretrained(name)
model.to(DEVICE)

# Load and preprocess audio
def load_audio(file_path, sampling_rate=16000):
    audio, sr = librosa.load(file_path, sr=sampling_rate)
    return audio, sr

speech_array, sampling_rate = load_audio(INPUT_AUDIO_PATH)
input_values = processor(speech_array, sampling_rate=sampling_rate, return_tensors="pt", padding=True).input_values.to(DEVICE)

# Encode target text
target_ids = processor.tokenizer(TARGET_TEXT, return_tensors="pt").input_ids.to(DEVICE)
print("===============================")
print(target_ids)

# Initialize perturbation
perturbation = torch.zeros_like(input_values, requires_grad=True, device=DEVICE)

# Optimize perturbation
optimizer = torch.optim.Adam([perturbation], lr=lr)

for step in tqdm(range(total_steps)):
    perturbed_input = input_values + perturbation  # Add perturbation
    perturbed_input = torch.clamp(perturbed_input, -1.0, 1.0)  # Clamp range

    outputs = model(perturbed_input)  # Run inference on perturbed input
    log_probs = torch.nn.functional.log_softmax(outputs.logits, dim=-1)
    log_probs = log_probs.permute(1, 0, 2)

    input_lengths = torch.tensor([log_probs.size(0)] * log_probs.size(1), dtype=torch.long, device=DEVICE)
    target_lengths = torch.full((target_ids.size(0),), target_ids.size(1), dtype=torch.long, device=DEVICE)

    # Compute loss
    loss_ctc = torch.nn.functional.ctc_loss(log_probs, target_ids, input_lengths, target_lengths)
    perturbation_regularization = torch.norm(perturbation, p=2)
    regularization_weight = rw
    loss = loss_ctc + regularization_weight * perturbation_regularization

    # Backprop and update perturbation
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_([perturbation], max_norm=1.0)
    optimizer.step()

    # Clamp perturbation range
    with torch.no_grad():
        perturbation.data = torch.clamp(perturbation.data, -EPSILON, EPSILON)

    # Print intermediate results
    if step % 10 == 0:
        perturbed_transcription = processor.decode(torch.argmax(outputs.logits, dim=-1)[0])
        print(f"Step {step}, Loss: {loss.item()}, Perturbed Transcription: {perturbed_transcription}")

# Final transcription
with torch.no_grad():
    perturbed_input = input_values + perturbation
    perturbed_input = torch.clamp(perturbed_input, -1.0, 1.0)
    outputs = model(perturbed_input)
    perturbed_transcription = processor.decode(torch.argmax(outputs.logits, dim=-1)[0])

# Save perturbed audio
perturbed_audio = perturbed_input.cpu().detach().numpy().squeeze()
sf.write(OUTPUT_AUDIO_PATH, perturbed_audio, sampling_rate)

# Original transcription
original_transcription = processor.decode(torch.argmax(model(input_values).logits, dim=-1)[0])
print(f"Original Transcription: {original_transcription}")
print(f"Target Transcription: {TARGET_TEXT}")
print(f"Perturbed Transcription: {perturbed_transcription}")
print(f"Perturbed audio saved to: {OUTPUT_AUDIO_PATH}")

from difflib import SequenceMatcher
import re

def calculate_accuracy(predicted, target, level="char"):
    """
    Calculate accuracy between predicted and target text
    :param predicted: Transcribed text
    :param target: Target text
    :param level: "char" or "word" level accuracy
    """
    if level == "char":
        matcher = SequenceMatcher(None, predicted, target)
        accuracy = matcher.ratio() * 100
    elif level == "word":
        predicted_words = predicted.split()
        target_words = target.split()
        matcher = SequenceMatcher(None, predicted_words, target_words)
        accuracy = matcher.ratio() * 100
    else:
        raise ValueError("Invalid level! Use 'char' or 'word'.")
    return accuracy

def clean_transcription(transcription):
    """
    Clean transcription by removing unnecessary characters
    """
    return transcription.replace(" ", "").replace("|", "")

def normalize_text(text):
    """
    Normalize text by removing spaces and punctuation
    """
    return re.sub(r"[^a-zA-Z0-9]", "", text)

cleaned_target_text = normalize_text(TARGET_TEXT)
cleaned_perturbed_transcription = clean_transcription(perturbed_transcription)

# Calculate accuracy
char_accuracy = calculate_accuracy(cleaned_perturbed_transcription, cleaned_target_text, level="char")
word_accuracy = calculate_accuracy(cleaned_perturbed_transcription, cleaned_target_text, level="word")

print("======================================================================================")
print(f"Original Transcription: {original_transcription}")
print(f"Target Transcription (cleaned): {cleaned_target_text}")
print(f"Perturbed Transcription (cleaned): {cleaned_perturbed_transcription}")
print("======================================================================================")
print(f"Character-level Accuracy: {char_accuracy:.2f}%")
print(f"Word-level Accuracy: {word_accuracy:.2f}%")
print("======================================================================================")

# Plot and save spectrograms
def save_spectrogram(audio, sampling_rate, title, file_name):
    plt.figure(figsize=(10, 6))
    spectrogram = librosa.stft(audio)
    spectrogram_db = librosa.amplitude_to_db(np.abs(spectrogram), ref=np.max)
    librosa.display.specshow(spectrogram_db, sr=sampling_rate, x_axis="time", y_axis="hz", cmap="viridis")
    plt.colorbar(format="%+2.0f dB")
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()

save_spectrogram(speech_array, sampling_rate, "Original Audio Spectrogram", "original_audio_spectrogram.png")
save_spectrogram(perturbed_audio, sampling_rate, "Perturbed Audio Spectrogram", "perturbed_audio_spectrogram.png")

# SNR (Signal-to-Noise Ratio)
def calculate_snr_max_ratio(original_audio, adversarial_audio):
    min_length = min(len(original_audio), len(adversarial_audio))
    original_audio = original_audio[:min_length]
    adversarial_audio = adversarial_audio[:min_length]
    noise = adversarial_audio - original_audio
    max_signal = np.max(np.abs(original_audio))
    max_noise = np.max(np.abs(noise)) + 1e-10
    snr = 20 * np.log10(max_signal / max_noise)
    return snr

if speech_array.shape == perturbed_audio.shape:
    print("Original and perturbed audio have the same shape")
else:
    print("Original and adversarial audio shapes differ, SNR may be inaccurate.")

snr_value = calculate_snr_max_ratio(speech_array, perturbed_audio)
print(f"Signal-to-Noise Ratio (SNR): {snr_value:.2f} dB")
print("Saved original audio spectrogram as 'original_audio_spectrogram.png'")
print("Saved perturbed audio spectrogram as 'perturbed_audio_spectrogram.png'")
print("============================================")

# PESQ (Perceptual Evaluation of Speech Quality)
from pesq import pesq, PesqError

def normalize_audio(audio):
    return audio / np.max(np.abs(audio))

def calculate_pesq(original_audio, perturbed_audio, sampling_rate):
    try:
        original_audio = normalize_audio(original_audio)
        perturbed_audio = normalize_audio(perturbed_audio)
        pesq_score = pesq(sampling_rate, original_audio, perturbed_audio, 'wb')
        return pesq_score
    except PesqError as e:
        print(f"PESQ calculation error: {e}")
        return None

original_audio_np = speech_array if isinstance(speech_array, np.ndarray) else speech_array.cpu().detach().numpy()
perturbed_audio_np = perturbed_audio if isinstance(perturbed_audio, np.ndarray) else perturbed_audio.cpu().detach().numpy()

min_length = min(len(original_audio_np), len(perturbed_audio_np))
original_audio_np = original_audio_np[:min_length]
perturbed_audio_np = perturbed_audio_np[:min_length]

pesq_score = calculate_pesq(original_audio_np, perturbed_audio_np, sampling_rate)
if pesq_score is not None:
    print(f"PESQ Score: {pesq_score:.2f}")
else:
    print("Failed to calculate PESQ score.")





    