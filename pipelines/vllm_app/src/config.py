# my_llm_service/config.py

import os

# Model configuration
MODEL_NAME = os.getenv("MODEL_NAME", "gpt2")
TENSOR_PARALLEL_SIZE = int(os.getenv("TENSOR_PARALLEL_SIZE", "0"))
RESPONSE_ROLE = os.getenv("RESPONSE_ROLE", "assistant")

# Ray Serve configuration
AUTOSCALING_CONFIG = {
    "min_replicas": int(os.getenv("MIN_REPLICAS", "2")),
    "max_replicas": int(os.getenv("MAX_REPLICAS", "2")),
    "target_ongoing_requests": int(os.getenv("TARGET_ONGOING_REQUESTS", "5")),
}
MAX_ONGOING_REQUESTS = int(os.getenv("MAX_ONGOING_REQUESTS", "1"))

# API configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000/v1")
API_KEY = os.getenv("API_KEY", "NOT A REAL KEY")

DEVICE = "cpu"  # or "cuda"