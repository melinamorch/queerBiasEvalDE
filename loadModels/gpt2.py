from transformers import AutoTokenizer, AutoModelWithLMHead

# Download and cache GPT-2 model & tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelWithLMHead.from_pretrained("gpt2")

model.save_pretrained("gpt2")  # Saves config.json, pytorch_model.bin, etc.
tokenizer.save_pretrained("gpt2")