# import all necessary libraries
import torch
from torch.profiler import profile, ProfilerActivity
import intel_extension_for_pytorch
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import BertModel

import intel_extension_for_pytorch as ipex
import time


import logging
import os
from datetime import datetime



# Get the current date and time
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

# Create a log file name using the current time
log_file_name = f"log_{current_time}.log"

# Example: Saving the log file in a specific directory
log_file_path = os.path.join('/workspaces/trint/', log_file_name)

os.environ['TORCH_LOGS_OUT'] = log_file_path


torch._inductor.config.trace.enabled = True
# torch._inductor.config.max_autotune_gemm_backends = "TRITON,ATEN"
# torch._inductor.config.max_autotune = True
torch._logging.set_logs(dynamo=logging.DEBUG,inductor=logging.DEBUG,schedule=False,output_code=False,fusion=False)



# Define the number of output labels for classification (e.g., 2 for binary classification)
output_size = 2  # Modify this based on your specific task

# Load pre-trained BERT model and tokenizer
# model_name = "bert-base-uncased"  # You can replace this with another pre-trained model if needed
# tokenizer = BertTokenizer.from_pretrained(model_name)
# model = BertForSequenceClassification.from_pretrained(model_name, num_labels=output_size)
# print(model.config)
# exit()

model = BertModel.from_pretrained("bert-base-uncased")
model.eval()

vocab_size = model.config.vocab_size
batch_size = 1
seq_length = 512
data = torch.randint(vocab_size, size=[batch_size, seq_length])

# Force model to use CPU
device = 'xpu:0'
model = model.to(device)

model = model.to("xpu")
data = data.to("xpu")
# model = ipex.optimize(model)
# Example usage - Text input to classify
# text = "This is an example sentence for classification."

# Tokenize the input text
# inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)

# Move the inputs to CPU
# inputs = {key: value.to(device) for key, value in inputs.items()}

# model(data)
# compile model
compiled_model = torch.compile(model, options={"freezing": True})

# model = ipex.optimize(model)

# outputs = compiled_model(data)


# Warm-up to avoid measuring overhead
with torch.no_grad():
    for _ in range(10):
        compiled_model(data)



with torch.no_grad():
    # with torch.xpu.amp.autocast(dtype=torch.float16):
        with profile(activities=[ProfilerActivity.CPU,
                                ProfilerActivity.XPU])as prof:
            # do what you want to profile here after the `with` statement with proper indent
            # Forward pass - inference
            start_time = time.time()
            outputs = compiled_model(data)
            end_time = time.time()

print(prof.key_averages().table())

inference_time = end_time - start_time
print(f"Inference time (compiled model): {inference_time:.6f} seconds")

prof.export_chrome_trace("trace_file_BERT_xpu_compile.json")


with torch.no_grad():
    # with torch.xpu.amp.autocast(dtype=torch.float16):
        with profile(activities=[ProfilerActivity.CPU,
                         ProfilerActivity.XPU],
                         record_shapes=True, profile_memory=True, with_stack=True, with_flops=True, with_modules=True) as prof0:
            # do what you want to profile here after the `with` statement with proper indent
            # Forward pass - inference
            outputs = compiled_model(data)

prof0.export_chrome_trace("trace_file_BERT_xpu_compile_detail.json")