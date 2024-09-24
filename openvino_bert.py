import torch
from transformers import BertModel
import onnx
import numpy as np
import openvino as ov
import time
from openvino.runtime import Core

# Initialize OpenVINO Core object
core = Core()

# Get available devices
devices = core.available_devices

# Print all available devices
print("Available devices:")
for device in devices:
    print(device)

############# Prepare PyTorch Model ###############
# Load the pretrained BERT model
model = BertModel.from_pretrained("bert-base-uncased")
model.eval()


# Define input data
vocab_size = model.config.vocab_size
batch_size = 1
seq_length = 512
data = torch.randint(vocab_size, size=[batch_size, seq_length])

############# Export the Model to ONNX ###############
onnx_model_path = "bert_model.onnx"
torch.onnx.export(
    model,                            # PyTorch model
    data,                             # Input to the model
    onnx_model_path,                  # Path to save the ONNX model
    input_names=["input_ids"],         # Input names
    output_names=["output"],           # Output names
    opset_version=14,                  # Use opset_version=14 for latest operators support
    dynamic_axes={"input_ids": {0: "batch_size", 1: "seq_length"}}  # Dynamic axes for dynamic input
)

print(f"Model exported to {onnx_model_path}")

compiled_model = core.compile_model(onnx_model_path, "GPU:0")

infer_request = compiled_model.create_infer_request()

import numpy as np

# Parameters
vocab_size = 30522  # Example vocab size for BERT; you can get this from the model config
batch_size = 1
seq_length = 512

# Generate random input data as NumPy array (OpenVINO expects NumPy arrays)
data = np.random.randint(0, vocab_size, size=(batch_size, seq_length), dtype=np.int64)

# Example: Printing the generated input
print("Input data shape:", data.shape)

# Create tensor from external memory
# input_tensor = ov.Tensor(array=memory, shared_memory=True)
# Set input tensor for model with one input
# Measure inference time
start_time = time.time()

# Start asynchronous inference
infer_request.start_async()

# Wait for the inference to complete
infer_request.wait()

# Measure end time
end_time = time.time()

# Get output tensor for model with one output
output = infer_request.get_output_tensor()
output_buffer = output.data

# Calculate inference time
inference_time = end_time - start_time
print(f"Inference Time with OpenVINO (async): {inference_time:.6f} seconds")

# Print output
print("Output:", output_buffer)
print("Execution finished")
