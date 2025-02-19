from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

model_name = "google/gemma-2b"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load model with MPS (Apple GPU) or CPU fallback
device = "mps" if torch.backends.mps.is_available() else "cpu"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",  # Let accelerate automatically manage the device
    torch_dtype=torch.float16  # Use reduced precision for better performance
)

# Create a text generation pipeline WITHOUT the `device` argument
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, truncation=False)

# Generate a response
input_text = "What is healthcare sector. Give the response in 250 words"
output = pipe(input_text, max_length=250, do_sample=True, temperature=0.7)

# Print response
print("Generated Response:\n", output[0]["generated_text"])
