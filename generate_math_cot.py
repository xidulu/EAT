from typing import Any, Dict
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import sys
import json
from tqdm import tqdm

from transformers import modeling_utils
import subprocess
import os
from dataset_utils import load_math500, load_aime2025, load_gpqa, load_humaneval, load_gpqa_mc, load_aime2025_2
from llm_utils import greedy_decode_config, deepseek_decode_config, model_name_mapper

import argparse
from transformers import HfArgumentParser

dataset_mapping = {
    "math500": load_math500,
    "aime2025": load_aime2025,
    "aime2025_2": load_aime2025_2,
    "gpqa": load_gpqa,
    'human_eval': load_humaneval,
    'gpqa_mc': load_gpqa_mc,
}



def recipe_main(config):
    seed = config.seed
    _model_name = config.model
    model = model_name_mapper[_model_name]
    do_sample = config.do_sample
    dataset_name = config.dataset_name
    save_dir = config.save_dir
    if do_sample:
        decode_config = deepseek_decode_config
    else:
        decode_config = greedy_decode_config

    if not hasattr(modeling_utils, "ALL_PARALLEL_STYLES") or modeling_utils.ALL_PARALLEL_STYLES is None:
        modeling_utils.ALL_PARALLEL_STYLES = ["tp", "none","colwise",'rowwise']

    print('Entering main function')
    print(f"GPU Memory Usage: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    device = "cuda"
    # device='auto'
    torch.manual_seed(seed)

    # Load model
    model_name = model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    # D = load_math500()
    D = dataset_mapping[dataset_name]()
    model_outputs = []
    # N = 20000
    N = 10000
    for msg in tqdm(D):
        text = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=N,
                return_dict_in_generate=True,
                **decode_config,
            )
        generated_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        model_outputs.append({'response' : generated_text})

    # Save the generated text to a file
    save_file_name = f"{_model_name}_{seed}_llm_output_{dataset_name}"
    if do_sample:
        save_file_name += "_do_sample"
    if N == 20000:
        save_file_name += "output_long_long.json"
    else:
        save_file_name += "output_long.json"
    local_save_dir = save_dir
    local_save_dir = os.path.join(local_save_dir, save_file_name)
    with open(local_save_dir, "w") as f:
        for output in model_outputs:
            f.write(json.dumps(output) + "\n")
    print(f"Generated text saved to {local_save_dir}")


def main() -> None:
    """
    Entrypoint that sets up the execution env: single-process or distributed.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='qwen3_new', help='model name')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--do_sample', action='store_true', help='whether to do sampling')
    parser.add_argument('--dataset_name', type=str, default='math500', help='dataset name')
    parser.add_argument('--save_dir', type=str, default='./reasoning_chains/', help='directory to save generated rollouts')
    args = parser.parse_args()
    config = args
    print(config)
    recipe_main(config)

if __name__ == "__main__":
    sys.exit(main())
