# setup.py
# vllm-app start

from setuptools import setup, find_packages

setup(
    name="vllm_app",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "ray",
        "fastapi",
        "uvicorn",
        "openai",
        "vllm",
        "click",
    ],
    entry_points={
        "console_scripts": [
            "vllm-app=src.cli:cli",
        ],
    },
)