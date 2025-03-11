import torch
from vllm import LLM, SamplingParams

# Initialize LLM with explicit CPU configuration
try:
    llm = LLM(model="gpt2", device="cpu", dtype="torch.float16", enforce_eager=True)
    sampling_params = SamplingParams(max_tokens=50)  # Short output
    prompt = "This is a test. The next word is"
    output = llm.generate(prompt, sampling_params)
    print(output[0].outputs[0].text)

except Exception as e:
    print(f"Error during LLM execution: {e}")
