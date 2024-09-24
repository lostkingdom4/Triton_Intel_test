import torch
from transformers import BertModel
import time
from openvino.runtime import Core  # Import OpenVINO Runtime

############# Prepare PyTorch Model ###############
# Load pretrained BERT model and set it to eval mode
model = BertModel.from_pretrained("bert-base-uncased")
model.eval()

# Define input data
vocab_size = model.config.vocab_size
batch_size = 1
seq_length = 512
data = torch.randint(vocab_size, size=[batch_size, seq_length])

############# Convert PyTorch Model to OpenVINO ###############
# Trace the model to generate a TorchScript representation
traced_model = torch.jit.trace(model, data)

# Save the TorchScript model to a file
torch_script_model_path = "bert_model_traced.pt"
traced_model.save(torch_script_model_path)

# Alternatively, you could export to ONNX and convert it using OpenVINO's Model Optimizer
# torch.onnx.export(model, data, "bert_model.onnx", opset_version=11)

############# Load the Model in OpenVINO ###############
# Initialize OpenVINO's Core
core = Core()

# Compile model using OpenVINO
model_ir = core.compile_model(torch_script_model_path, "CPU")

# Create an inference request object for asynchronous or synchronous inference
infer_request = model_ir.create_infer_request()

# Prepare the input data (note: OpenVINO uses NumPy arrays for input data)
import numpy as np
input_data = data.numpy()

############# Run Inference with OpenVINO ###############
# Perform inference and measure time
start_time = time.time()

# Synchronous inference
results = infer_request.infer({0: input_data})

end_time = time.time()

# Calculate and print inference time
inference_time = end_time - start_time
print(f"Inference Time with OpenVINO: {inference_time:.6f} seconds")

# Post-process results if needed (depends on the model's output format)
print("Execution finished")