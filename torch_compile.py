# TODO: source /opt/intel/oneapi/compiler/latest/env/vars.sh 

import torch
import intel_extension_for_pytorch
from transformers import BertModel
import time


import logging
import os
from datetime import datetime

# # Get the current date and time
# current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

# # Create a log file name using the current time
# log_file_name = f"log_{current_time}.log"

# # Example: Saving the log file in a specific directory
# log_file_path = os.path.join('/workspaces/trint/', log_file_name)

# os.environ['TORCH_LOGS_OUT'] = log_file_path


# torch._inductor.config.trace.enabled = True
# torch._inductor.config.max_autotune_gemm_backends = "TRITON,ATEN"
# torch._inductor.config.max_autotune = True
# torch._logging.set_logs(dynamo=logging.DEBUG,inductor=logging.DEBUG,schedule=False,output_code=False,fusion=False)


model = BertModel.from_pretrained("bert-base-uncased")
model.eval()
model = model.to("xpu")

vocab_size = model.config.vocab_size
batch_size = 1
seq_length = 512
data = torch.randint(vocab_size, size=[batch_size, seq_length])
data = data.to("xpu")


# compile model
compiled_model = torch.compile(model, backend="inductor",options={"freezing": True})
with torch.no_grad():
    output = compiled_model(data)

exit()

# inference main
# with torch.no_grad():
#     with torch.xpu.amp.autocast(dtype=torch.float16):
#         start_time = time.time()  # Start timing
#         output = compiled_model(data)
# torch.xpu.synchronize()  # Ensure all operations are done before measuring time
# end_time = time.time()  # End timing
# inference_time = end_time - start_time
# print(f"Inference Time: {inference_time:.6f} seconds")
# print("Execution finished")