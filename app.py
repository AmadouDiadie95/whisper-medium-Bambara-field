import os
from openvino.inference_engine import IECore
import soundfile as sf
import numpy as np

# Load the IR model and initialize the Inference Engine
model_xml = "./models/pytorch_model.xml"
model_bin = os.path.splitext(model_xml)[0] + ".bin"
ie = IECore()
net = ie.read_network(model=model_xml, weights=model_bin)
exec_net = ie.load_network(network=net, device_name="CPU")

# Load and preprocess the audio file
audio_file = "file.wav"
audio, sample_rate = sf.read(audio_file)
audio = audio.astype(np.float32)

# Ensure the audio is of appropriate length (16000 samples for Whisper)
if len(audio) != 16000:
    raise ValueError("Audio file must have 16000 samples for Whisper model.")

# Prepare the input blob
input_blob = next(iter(net.input_info))
input_data = {input_blob: np.expand_dims(audio, axis=0)}

# Perform inference
output = exec_net.infer(input_data)

# Process the output (this depends on the specifics of your Whisper model)
# You need to check the model's documentation or inspect its output shape to know how to process it.

# Example if the output is a classification result
output_blob = next(iter(net.outputs))
result = output[output_blob]

# You can then use 'result' for further processing or analysis

# Clean up
del exec_net
ie.deallocate_buffers()
