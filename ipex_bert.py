# import all necessary libraries
import torch
from torch.profiler import profile, ProfilerActivity
import intel_extension_for_pytorch
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import BertModel

import intel_extension_for_pytorch as ipex

log_level = 2
torch.xpu.set_log_level(log_level)
print(torch.xpu.get_log_level())
# log_path="./ipex.log"
# torch.xpu.set_log_output_file_path(log_path)
print(torch.xpu.get_log_output_file_path())
print(torch.xpu.get_log_component())
log_component = 'OPS'
torch.xpu.set_log_component(log_component)

model = BertModel.from_pretrained("bert-base-uncased")
model.eval()

vocab_size = model.config.vocab_size
batch_size = 1
seq_length = 512
data = torch.randint(vocab_size, size=[batch_size, seq_length])

# Force model to use CPU
device = 'xpu:0'
# model = model.to(device)

model = model.to("xpu")
data = data.to("xpu")

# compiled_model = torch.compile(model, options={"freezing": True})

model = ipex.optimize(model)

# Warm-up to avoid measuring overhead
with torch.no_grad():
    for _ in range(10):
        model(data)

with torch.no_grad():
    # with torch.xpu.amp.autocast(dtype=torch.float16):
        with profile(activities=[ProfilerActivity.CPU,
                                ProfilerActivity.XPU])as prof:
            # do what you want to profile here after the `with` statement with proper indent
            # Forward pass - inference
            outputs = model(data)


print(prof.key_averages().table())

prof.export_chrome_trace("trace_file_BERT_xpu_ipex.json")


with torch.no_grad():
    # with torch.xpu.amp.autocast(dtype=torch.float16):
        with profile(activities=[ProfilerActivity.CPU,
                         ProfilerActivity.XPU],
                         record_shapes=True, profile_memory=True, with_stack=True, with_flops=True, with_modules=True) as prof0:
            # do what you want to profile here after the `with` statement with proper indent
            # Forward pass - inference
            outputs = model(data)

prof0.export_chrome_trace("trace_file_BERT_xpu_ipex_detail.json")