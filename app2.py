from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torchaudio
from ctcdecode import CTCBeamDecoder

# Load the processor and model
processor = Wav2Vec2Processor.from_pretrained("models")
model = Wav2Vec2ForCTC.from_pretrained("models")

# Set to evaluation mode
model.eval()

# Load and process audio
waveform, sample_rate = torchaudio.load("file.wav")
input_values = processor(waveform.squeeze().numpy(), return_tensors="pt").input_values

# Perform inference
with torch.no_grad():
    logits = model(input_values).logits

# Apply CTC decoding
beam_decoder = CTCBeamDecoder(
    processor.model.config.vocab_size,
    beam_width=10,
    log_probs_input=True,
)
beam_results, beam_scores, timesteps, out_lens = beam_decoder.decode(logits)

# Get the best result
best_result = beam_results[0][0].numpy().tolist()

# Convert to text
predicted_ids = processor.convert_ids_to_tokens(best_result)
transcription = processor.decode(predicted_ids)

# Print the result
print("Transcription:", transcription)
