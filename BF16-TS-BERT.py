import torch
from transformers import BertModel
import time

############# code changes ###############
import intel_extension_for_pytorch as ipex

############# code changes ###############

model = BertModel.from_pretrained("bert-base-uncased")
model.eval()

vocab_size = model.config.vocab_size
batch_size = 1
seq_length = 512
data = torch.randint(vocab_size, size=[batch_size, seq_length])

#################### code changes #################
model = model.to("xpu")
data = data.to("xpu")
model = ipex.optimize(model, dtype=torch.bfloat16)
#################### code changes #################

with torch.no_grad():
    d = torch.randint(vocab_size, size=[batch_size, seq_length])
    ############################# code changes #####################
    d = d.to("xpu")
    with torch.xpu.amp.autocast(enabled=True, dtype=torch.bfloat16):
    ############################# code changes #####################
        model = torch.jit.trace(model, (d,), strict=False)
        model = torch.jit.freeze(model)
        start_time = time.time()  # Start timing
        model(data)
torch.xpu.synchronize()  # Ensure all operations are done before measuring time

end_time = time.time()  # End timing

inference_time = end_time - start_time
print(f"Inference Time: {inference_time:.6f} seconds")

print("Execution finished")