# import all necessary libraries
import torch
from torch.profiler import profile, ProfilerActivity
import intel_extension_for_pytorch
import torch.nn as nn


import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Define the number of output labels for classification (e.g., 2 for binary classification)
output_size = 2  # Modify this based on your specific task

# Load pre-trained BERT model and tokenizer
model_name = "bert-base-uncased"  # You can replace this with another pre-trained model if needed
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=output_size)

# Force model to use CPU
device = 'xpu:0'
model = model.to(device)

# Example usage - Text input to classify
text = "This is an example sentence for classification."

# Tokenize the input text
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)

# Move the inputs to CPU
inputs = {key: value.to(device) for key, value in inputs.items()}

# input_size = 10   # Input dimension
# hidden_size = 20  # Hidden layer dimension
# output_size = 5   # Output dimension


# # Define the MLP model
# class SimpleMLP(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(SimpleMLP, self).__init__()
#         self.fc1 = nn.Linear(input_size, hidden_size)  # First fully connected layer
#         self.relu = nn.ReLU()  # Activation function
#         self.fc2 = nn.Linear(hidden_size, output_size)  # Second fully connected layer

#     def forward(self, x):
#         out = self.fc1(x)
#         out = self.relu(out)
#         out = self.fc2(out)
#         return out

# model = SimpleMLP(input_size, hidden_size, output_size)

# model = model.to('xpu:0')


# these lines won't be profiled before enabling profiler tool
# input_tensor = torch.randn(1024, dtype=torch.float32, device='xpu:0')

# Create a random input tensor
# x = torch.randn(1, input_size, device='xpu:0')

# Forward pass
# output = model(x)
# print(output)

# Warm-up to avoid measuring overhead
with torch.no_grad():
    for _ in range(10):
        model(**inputs)


# enable Kineto supported profiler tool with a `with` statement
with profile(activities=[ProfilerActivity.CPU,
                         ProfilerActivity.XPU],
                         record_shapes=True, profile_memory=True, with_stack=False, with_flops=True, with_modules=True) as prof:
    # do what you want to profile here after the `with` statement with proper indent
    # Forward pass - inference
    with torch.no_grad():
        outputs = model(**inputs)
    # pass

# with profile(activities=[ProfilerActivity.CPU,
#                          ProfilerActivity.XPU],) as prof:
#     # do what you want to profile here after the `with` statement with proper indent
#     output_tensor_1 = torch.nonzero(input_tensor)
#     output_tensor_2 = torch.unique(input_tensor)

# print the result table formatted by the profiler tool as your wish
print(prof.key_averages().table())

prof.export_chrome_trace("trace_file_BERT_xpu_detail.json")


with profile(activities=[ProfilerActivity.CPU,
                         ProfilerActivity.XPU])as prof:
    # do what you want to profile here after the `with` statement with proper indent
    # Forward pass - inference
    with torch.no_grad():
        outputs = model(**inputs)

prof.export_chrome_trace("trace_file_BERT_xpu.json")