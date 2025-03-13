# vllm_app/cli.py

import click
from .llm import build_app
from .query import query_llm
from ray import serve

@click.group()
def cli():
    """My LLM Service CLI."""
    pass

@cli.command()
def start():
    """Start the LLM service."""
    cli_args = {
        "model": "gpt2",
        "tensor_parallel_size": 1,
        "response_role": "assistant",
    }
    app = build_app(cli_args)
    serve.run(app)

@cli.command()
def query():
    """Query the LLM service."""
    query_llm()

if __name__ == "__main__":
    cli()