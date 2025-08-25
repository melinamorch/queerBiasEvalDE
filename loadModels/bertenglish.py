from transformers import AutoTokenizer, AutoModelForMaskedLM

# Download English BERT from HuggingFace
model_name = "bert-base-cased"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)

# Save it locally inside the working directory
model.save_pretrained("english_bert")  # Saves config.json, pytorch_model.bin, etc.
tokenizer.save_pretrained("english_bert")