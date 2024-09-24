import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Define the number of output labels for classification (e.g., 2 for binary classification)
output_size = 2  # Modify this based on your specific task

# Load pre-trained BERT model and tokenizer
model_name = "bert-base-uncased"  # You can replace this with another pre-trained model if needed
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=output_size)

# Force model to use CPU
device = torch.device("cpu")
model = model.to(device)

# Example usage - Text input to classify
text = "This is an example sentence for classification."

# Tokenize the input text
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)

# Move the inputs to CPU
inputs = {key: value.to(device) for key, value in inputs.items()}

# Forward pass - inference
with torch.no_grad():
    outputs = model(**inputs)

# The outputs are logits, you can use softmax to get probabilities
logits = outputs.logits
probabilities = torch.softmax(logits, dim=-1)

print(f"Logits: {logits}")
print(f"Probabilities: {probabilities}")
# Print each encoder layer in the BERT model

bert_encoder = model.bert.encoder

print("BERT Encoder Layers:")
for i, layer in enumerate(bert_encoder.layer):
    print(f"Layer {i+1}:")
    print(layer)