from typing import Any, Dict
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import manta
import sys
import json
from tqdm import tqdm

from transformers import modeling_utils
import subprocess
import os
from utils import aws_s3_sync, launch_simple_ray_trainers, ExpConfig
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


model_dir = 's3://netflix-dataoven-prod-users/xiw/model_hub/models--deepseek-ai--DeepSeek-R1-0528-Qwen3-8B/'
s3_output_dir = 's3://netflix-dataoven-prod-users/xiw/llm_eval_output_math/'


def recipe_main(sub_id, run_local, config):
    seed = config.seed
    _model_name = config.model
    model = model_name_mapper[_model_name]
    do_sample = config.do_sample
    dataset_name = config.dataset_name
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
    N = 20000
    # N = 10000
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
    output_folder = f"{_model_name}_{seed}_llm_output_{dataset_name}"
    if do_sample:
        output_folder += "_do_sample"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    if N == 20000:
        output_file_name = "output_long_long.json"
    else:
        output_file_name = "output_long.json"
    local_save_dir = os.path.join(output_folder, output_file_name)
    s3_save_dir = os.path.join(s3_output_dir, output_folder)
    with open(local_save_dir, "w") as f:
        for output in model_outputs:
            f.write(json.dumps(output) + "\n")
    print(f"Generated text saved to {local_save_dir}")
    print(f"Syncing output to S3: {s3_save_dir}")
    # with open(local_save_dir, "w") as f:
    #     f.write(generated_text)
    aws_s3_sync(output_folder, s3_save_dir, only_show_errors=False)
        


def main() -> None:
    """
    Entrypoint that sets up the execution env: single-process or distributed.
    """
    parser = argparse.ArgumentParser()
    # The launcher explicitly provides the following args.
    parser.add_argument("--num_gpu", type=int, default=1)
    parser.add_argument("--sub_id", type=str)
    parser.add_argument("--test",
                        default=False,
                        action=argparse.BooleanOptionalAction)
    parser.add_argument(
        "--gpu_type", type=str, choices=["A100", "A100_80GB", "H200"], default="A100"
    )
    parser.add_argument("--run_local", action="store_true", default=False)
    args, unknown_args = parser.parse_known_args()
    print(unknown_args)

    parser = HfArgumentParser(ExpConfig)
    config, unknown_args = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    config.add_unknown_args(unknown_args)
    print(config)

    if not args.sub_id:
        raise Exception("Provide a sub_id via --sub_id!")
    if args.run_local:
        print("Running in local mode, no distributed setup.")
        # In local mode, we can just run the recipe directly.
        recipe_main(args.sub_id, args.run_local, config)
        return
    resources_per_worker = manta.resources(args.gpu_type, args.num_gpu)
    # Store unknown args in the config

    launch_simple_ray_trainers(
        resources_per_worker=resources_per_worker,
        n_gpu = args.num_gpu,
        train_func=recipe_main,
        sub_id=args.sub_id,
        run_local=args.run_local,
        config=config,
    )


if __name__ == "__main__":
    sys.exit(main())
