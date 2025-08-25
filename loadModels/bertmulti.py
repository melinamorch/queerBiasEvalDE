from transformers import AutoTokenizer, AutoModelForMaskedLM

# Download Multilingual BERT from HuggingFace
model_name = "bert-base-multilingual-cased"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)

# Save it locally inside the working directory
model.save_pretrained("multi_bert")  # Saves config.json, pytorch_model.bin, etc.
tokenizer.save_pretrained("multi_bert")