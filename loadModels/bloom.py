from transformers import AutoTokenizer, AutoModelForCausalLM

# Choose a valid Bloom model checkpoint
model_name = "bigscience/bloom-560m"

# Download and cache
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Save locally
model.save_pretrained("bloom-560m")
tokenizer.save_pretrained("bloom-560m")
