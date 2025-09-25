# Compute the n gram entropy !
import warnings
warnings.simplefilter('ignore') # In any case, try to avoid warnings as much as possible.
warnings.simplefilter('ignore', SyntaxWarning) # Use this instead if you can limit the type of warning.


import json

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np
from transformers import modeling_utils, DynamicCache
from tqdm import tqdm
import os
from tqdm import tqdm
from generation_utils import (
    get_multi_completion_hf, get_multi_completion_vllm, compute_entropy, compute_bigram_entropy_vectorized,
    calculate_conditional_perplexity, calculate_conditional_perplexity_fast,
    compute_diversity,
    compute_entropy_ngram
)
import argparse

model_dict = {
    'dpsk_new': 'deepseek-ai/DeepSeek-R1-0528-Qwen3-8B',
    'qwen_base': 'Qwen/Qwen3-8B',
    'qwen3_new': 'Qwen/Qwen3-4B-Thinking-2507',
    'dpsk_small': 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B',
    'dpsk_llama': 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B',
}

# Add command line argument parsing
parser = argparse.ArgumentParser(description="Math Evaluation with Entropy Calculation")
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--include_final_answer_str', action='store_true', default=False)
parser.add_argument('--model_solution_dir', type=str)
parser.add_argument('--eval_model', type=str)
parser.add_argument('--save_dir', type=str)
parser.add_argument('--save_name', type=str)
parser.add_argument('--qid_limit', type=int, default=-1)

args = parser.parse_args()
gpu_id = args.gpu
include_final_answer_str = args.include_final_answer_str
eval_model = args.eval_model
qid_limit = args.qid_limit

if not hasattr(modeling_utils, "ALL_PARALLEL_STYLES") or modeling_utils.ALL_PARALLEL_STYLES is None:
    modeling_utils.ALL_PARALLEL_STYLES = ["tp", "none","colwise",'rowwise']

if eval_model in model_dict:
    model_name = model_dict[eval_model]
else:
    raise ValueError(f"Unknown eval_model: {eval_model}. Available options are: {list(model_dict.keys())}")

device = f"cuda:{gpu_id}"
hf_model = AutoModelForCausalLM.from_pretrained(model_name)
hf_model.to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

save_dir = args.save_dir
save_name = args.save_name + '{}.npy'
model_solutions_dir = args.model_solution_dir

assert os.path.exists(model_solutions_dir), f"File {model_solutions_dir} does not exist!"

model_solutions = []
with open(model_solutions_dir, 'r') as f:
    for line in f:
        if line.strip():
            model_solutions.append(json.loads(line.strip()))

print('CoT loaded, number of problems: ', len(model_solutions))

if '0528' in model_name:
    num_newline = 1
else:
    num_newline = 2


all_entropies = []

print(model_name)

for qid, row in tqdm(
        (list(enumerate(model_solutions)))
    ):
    if (qid_limit > 0) and (qid >= qid_limit):
        break
    if 'response' in row:
        model_response = row['response']
    else:
        model_response = row
    model_response = model_response.split('</think>')[0]
    header = model_response.split("<think>")[0] + "<think>\n"
    reasoning_line = model_response.split("<think>")[1].split('\n\n')
    reasoning_line[0] = reasoning_line[0].replace('\n', '')
    reasoning_line[-1] = reasoning_line[-1].replace('\n', '')
    entropy = []
    cache = DynamicCache()
    for i in tqdm(range(len(reasoning_line))):
        if (i < len(reasoning_line)) and (len(reasoning_line[i]) == 0):
            continue
        partial_answer = (header + "\n\n".join(reasoning_line[:i + 1]))
        entropy_2 = compute_entropy(
            hf_model,
            tokenizer,
            partial_answer + "\n\n",
            cache
        )
        cache.crop(-1)

        if not include_final_answer_str:
            extra_str =  "\n</think>" + "\n" * num_newline
        else:
            num_newline = 1
            extra_str =  "\n</think>" + "\n" * num_newline + 'Final answer: '

        ngram_entropy = compute_entropy_ngram(
            hf_model,
            tokenizer,
            partial_answer + extra_str,
            # num_mc_sample=10,
            N=20,
            K=4,
            kv_cache=cache
        )
        if isinstance(ngram_entropy, float):
            ngram_entropy = [ngram_entropy]
        entropy.append(
            (entropy_2,) + tuple(ngram_entropy)
        )
    
    _save_name = save_name.format(qid)
    np.save(
        os.path.join(save_dir, _save_name), # New: With bigram entropy included
        entropy
    )
