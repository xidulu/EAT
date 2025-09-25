import json

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np
from transformers import modeling_utils
from tqdm import tqdm
from vllm import LLM, SamplingParams

from generation_utils import (
    get_multi_completion_vllm,
)

model_name = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"

vllm_model = LLM(
        model=model_name,
        enable_prefix_caching=True,
        max_model_len=40960
    )