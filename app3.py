from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import torchaudio
import numpy as np

# Load the processor and model
processor = Wav2Vec2Processor.from_pretrained("models")
model = Wav2Vec2ForCTC.from_pretrained("models")

# Set to evaluation mode
model.eval()

# Load and process audio
waveform, sample_rate = torchaudio.load("old-file.wav")
input_values = processor(waveform.squeeze().numpy(), return_tensors="pt").input_values

# Perform inference
with torch.no_grad():
    logits = model(input_values).logits

# Get the predicted IDs at each time step
predicted_ids = np.argmax(logits, axis=-1)[0].numpy()

# Convert to text
transcription = processor.decode(predicted_ids[0])

# Print the result
print("Transcription:", transcription)
