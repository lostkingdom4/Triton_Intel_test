import torch
import intel_extension_for_pytorch
from transformers import BertModel
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
torch._inductor.config.max_autotune_gemm_backends = "TRITON,ATEN"
torch._inductor.config.max_autotune = True
torch._logging.set_logs(dynamo=logging.DEBUG,inductor=logging.DEBUG,schedule=True,output_code=False,fusion=True)

class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(100, 10)

    def forward(self, x):
        return torch.nn.functional.relu(self.lin(x))

mod = MyModule().to('xpu')

# Generate two large matrices and move them to GPU
# a = torch.randn(10000, 10000).to('xpu')
# b = torch.randn(10000, 10000).to('xpu')

options = {'max_autotune':True}
# mod(torch.randn(10, 100))


opt_square = torch.compile(mod)
opt_square(torch.randn(10, 100).to('xpu'))