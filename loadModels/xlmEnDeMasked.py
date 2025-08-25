import torch
from transformers import XLMTokenizer, XLMWithLMHeadModel

tokenizer = XLMTokenizer.from_pretrained("xlm-mlm-ende-1024")
model = XLMWithLMHeadModel.from_pretrained("xlm-mlm-ende-1024")


model.save_pretrained("xlm-mlm-ende-1024")  # Saves config.json, pytorch_model.bin, etc.
tokenizer.save_pretrained("xlm-mlm-ende-1024")