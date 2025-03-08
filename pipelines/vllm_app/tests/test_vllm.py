# python test_vllm.py
from vllm.entrypoints.llm import LLM

llm = LLM(model="NousResearch/Meta-Llama-3-8B-Instruct")
output = llm.generate("Hello, how are you?")
print(output)