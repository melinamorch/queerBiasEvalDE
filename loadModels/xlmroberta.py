import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
model = AutoModelForMaskedLM.from_pretrained("xlm-roberta-base")


model.save_pretrained("xlm-roberta-base")  # Saves config.json, pytorch_model.bin, etc.
tokenizer.save_pretrained("xlm-roberta-base")