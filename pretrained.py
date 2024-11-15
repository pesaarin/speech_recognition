import pandas as pd
import torch
import torchaudio
import soundfile as sf
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import re
import os
device = "cuda" if torch.cuda.is_available() else "cpu"
df = pd.read_csv("geo_ASR_challenge_2024/dev.csv")

processor = Wav2Vec2Processor.from_pretrained("cpierse/wav2vec2-large-xlsr-53-esperanto")
model = Wav2Vec2ForCTC.from_pretrained("cpierse/wav2vec2-large-xlsr-53-esperanto").to(device)

chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"\“\%\‘\”\�\„\«\(\»\)\’\']'
def preprocess_text(text):
    return re.sub(chars_to_ignore_regex, '', text).lower()

resampler = torchaudio.transforms.Resample(orig_freq=48000, new_freq=16000)

def process_audio(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    speech_array, sampling_rate = sf.read(path)
    if sampling_rate != 16000:
        
        speech_array = resampler(torch.tensor(speech_array).unsqueeze(0)).squeeze(0).numpy()
    return speech_array

def evaluate_single(path, text):
    speech = process_audio(path)

    inputs = processor(
        speech, sampling_rate=16000, return_tensors="pt", padding=True
    ).to(device)

    with torch.no_grad():
        logits = model(
            inputs.input_values, attention_mask=inputs.attention_mask
        ).logits

    pred_ids = torch.argmax(logits, dim=-1)
    predicted_text = processor.batch_decode(pred_ids)[0]

    return predicted_text, text

for idx, row in df.head(100).iterrows():
    full_path = os.path.join("geo_ASR_challenge_2024", row["file"])
    full_path = os.path.normpath(full_path)
    
    try:
        predicted, reference = evaluate_single(full_path, row["transcript"])
        print(f"Row {idx + 1}: Prediction: {predicted} | Reference: {reference}")
    except Exception as e:
        print(f"Error processing row {idx + 1}: {e}")
