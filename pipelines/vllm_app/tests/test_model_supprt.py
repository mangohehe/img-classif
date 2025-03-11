import asyncio
from vllm.engine.async_llm_engine import AsyncLLMEngine, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from uuid import uuid4

def extract_final_text(request_outputs):
    """
    Given an iterable of RequestOutput objects, extract the final generated text.
    It will use the text from the first CompletionOutput in the RequestOutput
    where finished is True. If none are marked as finished, it will use the last one.
    """
    final_output = None
    for req in request_outputs:
        if req.finished:
            final_output = req
            break
    # If none finished, use the last one available.
    if final_output is None and request_outputs:
        final_output = request_outputs[-1]
    
    if final_output is not None and final_output.outputs:
        # Assuming the first output holds the desired text.
        return final_output.outputs[0].text
    else:
        return ""

async def test_generation():
    engine_args = AsyncEngineArgs(model="gpt2", device="cpu")
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    prompt = "You are a helpful assistant.\nUser: What are some highly rated restaurants in San Francisco?\nAssistant:"
    sampling_params = SamplingParams(max_tokens=100)
    request_id = str(uuid4())

    outputs_list = []
    async for req_out in engine.generate(prompt, sampling_params, request_id):
        outputs_list.append(req_out)
    
    final_text = extract_final_text(outputs_list)
    print("Final Generated output:")
    print(final_text)

asyncio.run(test_generation())