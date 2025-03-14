# my_llm_service/llm.py

from vllm.engine.async_llm_engine import SamplingParams
from starlette.responses import JSONResponse
from fastapi import Request
import asyncio
from typing import Dict, Optional, List
from fastapi import FastAPI
from starlette.requests import Request
from starlette.responses import StreamingResponse, JSONResponse
from ray import serve
from uuid import uuid4
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.cli_args import make_arg_parser
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm.engine.async_llm_engine import AsyncLLMEngine, SamplingParams
import time

from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ErrorResponse,
)
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_models import (
    BaseModelPath,
    LoRAModulePath,
    PromptAdapterPath,
    OpenAIServingModels,
)
from vllm.utils import FlexibleArgumentParser
from vllm.entrypoints.logger import RequestLogger

from .config import MODEL_NAME, TENSOR_PARALLEL_SIZE, RESPONSE_ROLE, AUTOSCALING_CONFIG, MAX_ONGOING_REQUESTS
from .utils import setup_logger

logger = setup_logger("ray.serve")

app = FastAPI()

@staticmethod
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
    if final_output is None and request_outputs:
        final_output = request_outputs[-1]

    if final_output is not None and final_output.outputs:
        return final_output.outputs[0].text
    else:
        return ""


@serve.deployment(
    autoscaling_config=AUTOSCALING_CONFIG,
    max_ongoing_requests=MAX_ONGOING_REQUESTS,
)
@serve.ingress(app)
class VLLMDeployment:
    def __init__(
        self,
        engine_args: AsyncEngineArgs,
        response_role: str,
        lora_modules: Optional[List[LoRAModulePath]] = None,
        prompt_adapters: Optional[List[PromptAdapterPath]] = None,
        request_logger: Optional[RequestLogger] = None,
        chat_template: Optional[str] = None,
    ):
        logger.info(f"Starting with engine args: {engine_args}")
        self.openai_serving_chat = None
        self.engine_args = engine_args
        self.response_role = response_role
        self.lora_modules = lora_modules
        self.prompt_adapters = prompt_adapters
        self.request_logger = request_logger
        self.chat_template = chat_template
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)

    @app.post("/v1/chat/completions")
    async def create_chat_completion(
        self, request: ChatCompletionRequest, raw_request: Request
    ):
        """Existing chat endpoint remains unchanged."""
        if not self.openai_serving_chat:
            model_config = await self.engine.get_model_config()
            models = OpenAIServingModels(
                self.engine,
                model_config,
                [
                    BaseModelPath(
                        name=self.engine_args.model, model_path=self.engine_args.model
                    )
                ],
                lora_modules=self.lora_modules,
                prompt_adapters=self.prompt_adapters,
            )
            self.openai_serving_chat = OpenAIServingChat(
                self.engine,
                model_config,
                models,
                self.response_role,
                request_logger=self.request_logger,
                chat_template=self.chat_template,
                chat_template_content_format="auto",
            )
        logger.info(f"Chat Request: {request}")
        generator = await self.openai_serving_chat.create_chat_completion(request, raw_request)
        if isinstance(generator, ErrorResponse):
            return JSONResponse(content=generator.model_dump(), status_code=generator.code)
        if request.stream:
            return StreamingResponse(content=generator, media_type="text/event-stream")
        else:
            assert isinstance(generator, ChatCompletionResponse)
            return JSONResponse(content=generator.model_dump())


    @app.post("/v1/completions")
    async def create_completion(self, raw_request: Request):
        try:
            payload = await raw_request.json()
        except Exception as e:
            logger.error(f"Invalid JSON: {e}")
            return JSONResponse(
                content={"error": "Invalid JSON payload", "details": str(e)},
                status_code=400,
            )

        prompt = payload.get("prompt")
        max_tokens = payload.get("max_tokens", 100)

        if prompt is None:
            return JSONResponse(
                content={"error": "Missing 'prompt' in request"},
                status_code=400,
            )

        logger.info(f"Completion Request: prompt={prompt}, max_tokens={max_tokens}")

        start_time = time.time()
        try:
            sampling_params = SamplingParams(max_tokens=max_tokens)
            request_id = str(uuid4())
            outputs_list = []
            async for req_out in self.engine.generate(prompt, sampling_params, request_id):
                current = time.time()
                logger.info(f"Received output after {current - start_time:.2f} seconds: {req_out}")
                outputs_list.append(req_out)
            final_text = extract_final_text(outputs_list)
            total = time.time() - start_time
            logger.info(f"Final text extracted after {total:.2f} seconds: {final_text}")
        except asyncio.TimeoutError:
            logger.error("Generation timed out")
            return JSONResponse(
                content={"error": "Generation timed out"},
                status_code=500,
            )
        except Exception as e:
            logger.error(f"Generation error: {repr(e)}")
            return JSONResponse(
                content={"error": "Generation failed", "details": repr(e)},
                status_code=500,
            )

        return JSONResponse(content={"text": final_text})


def parse_vllm_args(cli_args: Dict[str, str]):
    """Parses vLLM args based on CLI inputs."""
    arg_parser = FlexibleArgumentParser(
        description="vLLM OpenAI-Compatible RESTful API server."
    )
    parser = make_arg_parser(arg_parser)
    arg_strings = []
    for key, value in cli_args.items():
        arg_strings.extend([f"--{key}", str(value)])
    logger.info(arg_strings)
    parsed_args = parser.parse_args(args=arg_strings)
    return parsed_args


def build_app(cli_args: Dict[str, str]) -> serve.Application:
    """Builds the Serve app based on CLI arguments."""
    if "accelerator" in cli_args.keys():
        accelerator = cli_args.pop("accelerator")
    else:
        accelerator = "GPU"
    parsed_args = parse_vllm_args(cli_args)
    engine_args = AsyncEngineArgs.from_cli_args(parsed_args)
    engine_args.worker_use_ray = True

    tp = engine_args.tensor_parallel_size
    logger.info(f"Tensor parallelism = {tp}")
    
    # for i in range(tp):
    #    pg_resources.append({"CPU": 1, accelerator: 1})  # for the vLLM actors
    pg_resources = [{"CPU": 4}] * 1 
    return VLLMDeployment.options(
       num_replicas=1,placement_group_bundles=pg_resources, placement_group_strategy="SPREAD"
    ).bind(
        engine_args,
        parsed_args.response_role,
        parsed_args.lora_modules,
        parsed_args.prompt_adapters,
        cli_args.get("request_logger"),
        parsed_args.chat_template,
    )
