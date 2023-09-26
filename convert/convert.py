import torch
import torch.onnx

# Load the PyTorch model
model = torch.load("../models/pytorch_model.bin", map_location=torch.device('cpu'))

# Set to evaluation mode
model.eval()

# Create a dummy input (you need to know the input shape beforehand)
dummy_input = torch.randn(1, input_channels, input_height, input_width)

# Export the model to ONNX format
onnx_file = "whisper_model.onnx"
torch.onnx.export(model, dummy_input, onnx_file, opset_version=12)


# There is my trained files results in the /models folder :
# added_tokens.json
# all_results.json
# config.json
# eval_results.json
# merges.txt
# normalizer.json
# preprocessor_config.json
# pytorch_model.bin
# special_tokens_map.json
# tokenizer_config.json
# train_results.json
# trainer_state.json
# training_args.bin
# vocab.json