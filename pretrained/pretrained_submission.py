import pandas as pd
import torch
import torchaudio
import soundfile as sf
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import re
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

df = pd.read_csv("geo_ASR_challenge_2024/test_release.csv")

processor = Wav2Vec2Processor.from_pretrained("cpierse/wav2vec2-large-xlsr-53-esperanto")
model = Wav2Vec2ForCTC.from_pretrained("cpierse/wav2vec2-large-xlsr-53-esperanto").to(device)

resampler = torchaudio.transforms.Resample(orig_freq=48000, new_freq=16000)

def process_audio(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    speech_array, sampling_rate = sf.read(path)
    if sampling_rate != 16000:
        speech_array = resampler(torch.tensor(speech_array).unsqueeze(0)).squeeze(0).numpy()
    return speech_array

def predict_transcript(path):
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

    # Apply known fixes for common errors
    predicted_text = predicted_text.replace("k", "_").replace("g", "k").replace("_", "g")
    predicted_text = predicted_text.replace("i", "_").replace("e", "i").replace("_", "e")
    predicted_text = predicted_text.replace("f", "_").replace("v", "f").replace("_", "v")
    
    return predicted_text

predictions = []
for idx, row in df.iterrows():
    full_path = os.path.join("geo_ASR_challenge_2024", row["file"])
    full_path = os.path.normpath(full_path)
    
    predicted = predict_transcript(full_path)
    predictions.append(predicted)
    print(f"Row {idx + 1}: Prediction: {predicted}")


df['transcript'] = predictions

output_path = "geo_ASR_challenge_2024/test_release_predictions.csv"
df.to_csv(output_path, index=False)

print(f"Predictions saved to {output_path}")