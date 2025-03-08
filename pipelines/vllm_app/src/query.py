# my_llm_service/query.py

from openai import OpenAI
from .config import API_BASE_URL, API_KEY

def query_llm():
    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=API_KEY,
    )

    chat_completion = client.chat.completions.create(
        model="NousResearch/Meta-Llama-3-8B-Instruct",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": "What are some highly rated restaurants in San Francisco?'",
            },
        ],
        temperature=0.01,
        stream=True,
        max_tokens=100,
    )

    for chat in chat_completion:
        if chat.choices[0].delta.content is not None:
            print(chat.choices[0].delta.content, end="")