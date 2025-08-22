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
    # calculate_conditional_perplexity_with_kv_cache
)
import argparse

model_dict = {
    'dpsk_new': 'deepseek-ai/DeepSeek-R1-0528-Qwen3-8B',
    'qwen_base': 'Qwen/Qwen3-8B',
    'dpsk_small': 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B',
    'dpsk_llama': 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B',
}

# Add command line argument parsing
parser = argparse.ArgumentParser(description="Math Evaluation with Entropy Calculation")
parser.add_argument('--part', type=int, default=-1)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--is_qwen', action='store_true', default=False)
parser.add_argument('--is_qwen_with_sample', action='store_true', default=False)
parser.add_argument('--include_final_answer_str', action='store_true', default=False)
parser.add_argument('--seed', type=int, default=-1)
parser.add_argument('--dataset', type=str, default=None)
parser.add_argument('--model_solution_dir', type=str, default=None)
parser.add_argument('--eval_model', type=str, default=None)
parser.add_argument('--save_name', type=str, default=None)

args = parser.parse_args()
part = args.part
gpu_id = args.gpu
is_qwen = args.is_qwen
is_qwen_with_sample = args.is_qwen_with_sample
include_final_answer_str = args.include_final_answer_str
seed = args.seed
dataset = args.dataset
eval_model = args.eval_model


if part == -1:
    part_idx = list(range(500))
else:
    part_idx = list(range(part * 125, (part + 1) * 125))


if not hasattr(modeling_utils, "ALL_PARALLEL_STYLES") or modeling_utils.ALL_PARALLEL_STYLES is None:
    modeling_utils.ALL_PARALLEL_STYLES = ["tp", "none","colwise",'rowwise']

if is_qwen:
    model_name = "Qwen/Qwen3-4B"
elif is_qwen_with_sample:
    model_name = "Qwen/Qwen3-4B"
else:
    model_name = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"

if eval_model is not None:
    if eval_model in model_dict:
        model_name = model_dict[eval_model]
    else:
        raise ValueError(f"Unknown eval_model: {eval_model}. Available options are: {list(model_dict.keys())}")

device = f"cuda:{gpu_id}"
hf_model = AutoModelForCausalLM.from_pretrained(model_name)
hf_model.to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

save_dir = f'./math_outputs'

if args.model_solution_dir is not None:
    model_solutions_dir = args.model_solution_dir
else:
    if is_qwen:
        model_solutions_dir = "./model_solutions/qwen_new_no_sample_long.json"
    elif is_qwen_with_sample:
        model_solutions_dir = "./model_solutions/qwen_new_with_sample_long.json"
    else:
        model_solutions_dir = "./model_solutions/dpsk_new_with_sample_long.json"

    if seed != -1:
        model_solutions_dir = model_solutions_dir.replace('.json', f'_seed_{seed}.json')

    if dataset is not None:
        model_solutions_dir = model_solutions_dir.replace('.json', f'_{dataset}.json')

model_solutions = []
with open(model_solutions_dir, 'r') as f:
    for line in f:
        if line.strip():
            model_solutions.append(json.loads(line.strip()))


if '0528' in model_name:
    num_newline = 1
else:
    num_newline = 2


all_entropies = []

for qid, row in tqdm(
        (list(enumerate(model_solutions)))
    ):
    if qid not in part_idx:
        continue
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
            N=2,
            K=1,
            kv_cache=cache
        )
        if isinstance(ngram_entropy, float):
            ngram_entropy = [ngram_entropy]
        entropy.append(
            (entropy_2,) + tuple(ngram_entropy)
        )
        

    entropy = np.array(entropy)
    save_dir = '/root/tts_and_entropy/outputs/entropy_log_math'
    # np.save(
    #     os.path.join(save_dir, f"entropy_{qid}_new.npy"), # New: With bigram entropy included
    #     entropy
    # )

    if is_qwen:
        save_name = f"entropy_{qid}_new_qwen_new_no_sample_ngram.npy"
    elif is_qwen_with_sample:
        save_name = f"entropy_{qid}_new_qwen_new_with_sample_ngram.npy"
    else:
        save_name = f"entropy_{qid}_new_dpsk_new_with_sample_ngram.npy"

    if include_final_answer_str:
        save_name = save_name.replace('.npy', '_with_final_answer_str.npy')
    
    if seed != -1:
        save_dir = f'/root/tts_and_entropy/outputs/entropy_log_math_multiple_seeds/seed_{seed}'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # save_name = save_name.replace('.npy', f'_seed_{seed}.npy')

    if dataset is not None:
        save_name = save_name.replace('.npy', f'_{dataset}.npy')

    if args.eval_model is not None:
        save_name = save_name.replace('.npy', f'_eatEvaluedBy{args.eval_model}.npy')

    if args.save_name is not None:
        save_name = args.save_name.format(qid)

    np.save(
        os.path.join(save_dir, save_name), # New: With bigram entropy included
        entropy
    )

    # if is_qwen:
    #     np.save(
    #         os.path.join(save_dir, f"entropy_{qid}_new_qwen_new_no_sample_ngram.npy"), # New: With bigram entropy included
    #         entropy
    #     )
    # elif is_qwen_with_sample:
    #     np.save(
    #         os.path.join(save_dir, f"entropy_{qid}_new_qwen_new_with_sample_ngram.npy"), # New: With bigram entropy included
    #         entropy
    #     )
    # else:
    #     np.save(
    #         os.path.join(save_dir, f"entropy_{qid}_new_dpsk_new_with_sample_ngram.npy"), # New: With bigram entropy included
    #         entropy
    #     )
