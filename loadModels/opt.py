from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "facebook/opt-350m"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

model.save_pretrained("opt-350m")
tokenizer.save_pretrained("opt-350m")
