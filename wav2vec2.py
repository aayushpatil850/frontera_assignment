import librosa
import torch
from transformers import AutoProcessor, AutoModelForAudioClassification
import numpy as np
import torch.nn.functional as F  # For softmax

# Load the fine-tuned model and processor
model = AutoModelForAudioClassification.from_pretrained("./fine_tuned_wav2vec2")
processor = AutoProcessor.from_pretrained("./fine_tuned_wav2vec2")

# Mapping for labels
label_mapping = {0: 'crying', 1: 'screaming', 2: 'normal'}

# Function to predict the label of a single audio file
def predict_audio_label(audio_path):
    # Load audio file using librosa
    audio, sr = librosa.load(audio_path, sr=16000)

    # Preprocess the audio file (removed truncation argument)
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
    
    # Perform the prediction
    with torch.no_grad():
        logits = model(**inputs).logits

    # Convert logits to probabilities using softmax
    probs = F.softmax(logits, dim=-1)

    # Get the predicted label
    predicted_class_id = torch.argmax(logits, dim=-1).item()
    predicted_label = label_mapping[predicted_class_id]
    
    # Print the probabilities for all classes
    for i, label in label_mapping.items():
        print(f"Probability of {label}: {probs[0][i].item():.4f}")
    
    return predicted_label

# Test with a single audio file
audio_path = "/home/aayushjeevan/Desktop/FRONTERA/archive3/Screaming/0kQANiakiH0_out.wav"
predicted_label = predict_audio_label(audio_path)

print(f"The predicted label for the audio file is: {predicted_label}")
