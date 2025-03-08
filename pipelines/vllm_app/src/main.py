# main.py

import ray
from ray import serve
from config import MODEL_NAME, TENSOR_PARALLEL_SIZE, RESPONSE_ROLE
from llm import build_app

def main():
    # Initialize Ray
    ray.init(address="auto", num_cpus=10)

    # Define CLI arguments
    cli_args = {
        "model": MODEL_NAME,
        "tensor_parallel_size": TENSOR_PARALLEL_SIZE,
        "response_role": RESPONSE_ROLE,
        "accelerator": "CPU",
    }

    # Build and deploy the Serve application
    app = build_app(cli_args)
    serve.run(app)

if __name__ == "__main__":
    main()