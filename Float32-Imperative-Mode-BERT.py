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

######## code changes #######
model = model.to("xpu")
data = data.to("xpu")
model = ipex.optimize(model)
######## code changes #######

start_time = time.time()  # Start timing

with torch.no_grad():
    model(data)
torch.xpu.synchronize()  # Ensure all operations are done before measuring time

end_time = time.time()  # End timing

inference_time = end_time - start_time
print(f"Inference Time: {inference_time:.6f} seconds")

print("Execution finished")