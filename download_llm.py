# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

# Specify the custom cache directory
cache_dir = "/data/rech/huiyuche/huggingface"

# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct",cache_dir = cache_dir)
# model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", cache_dir = cache_dir)


tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2",cache_dir = cache_dir)
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2",cache_dir = cache_dir)