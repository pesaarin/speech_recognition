import pandas as pd
import torch
import torchaudio
import soundfile as sf
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import re
import os
from jiwer import wer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
df = pd.read_csv("geo_ASR_challenge_2024/dev.csv")

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

def prepare_confusion_matrix_data(predictions, references):
    all_references = []
    all_predictions = []
    
    for ref, pred in zip(references, predictions):
        ref_tokens = list(ref)
        pred_tokens = list(pred)
        
        # Pad to equal lengths
        max_len = max(len(ref_tokens), len(pred_tokens))
        ref_tokens += [""] * (max_len - len(ref_tokens))
        pred_tokens += [""] * (max_len - len(pred_tokens))
        
        all_references.extend(ref_tokens)
        all_predictions.extend(pred_tokens)
    
    return all_references, all_predictions

predictions = []
references = []
errors = []

for idx, row in df.iterrows():
    full_path = os.path.join("geo_ASR_challenge_2024", row["file"])
    full_path = os.path.normpath(full_path)
    
    
    predicted, reference = evaluate_single(full_path, row["transcript"])
    predictions.append(predicted)
    references.append(reference)
    print(f"Row {idx + 1}: Prediction: {predicted} | Reference: {reference}")


total_wer = wer(references, predictions)
print(f"Total WER for dataset: {total_wer:.2%}")

ref_tokens, pred_tokens = prepare_confusion_matrix_data(predictions, references)

labels = sorted(set(ref_tokens + pred_tokens))
cm = confusion_matrix(ref_tokens, pred_tokens, labels=labels)

plt.figure(figsize=(12, 10))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap="viridis", xticks_rotation="vertical")
plt.title("Token-Level Confusion Matrix")
plt.show()
