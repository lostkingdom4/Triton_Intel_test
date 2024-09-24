import gzip
import json
import torch

# Function to open and read a .json.gz file
def read_json_gz(file_path):
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        data = json.load(f)  # Load the JSON data
    return data

# Example usage
file_path = 'plot.raw.json.gz'  # Replace with your file path
data = read_json_gz(file_path)

# Print or process the data
print(data)
Categorys = [cat for cat in torch.profiler._memory_profiler.Category]
print(Categorys)