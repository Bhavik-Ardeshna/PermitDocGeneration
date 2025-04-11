import os

from together import Together

client = Together(
    api_key=os.getenv("TOGETHER_API_KEY"), base_url=os.getenv("INFERENCE_LLM_BASE_URL")
)
