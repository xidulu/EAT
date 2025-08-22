import json

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np
from transformers import modeling_utils
from tqdm import tqdm
from vllm import LLM, SamplingParams
import os
from tqdm import tqdm
from generation_utils import (
    # get_multi_completion_hf, 
    get_multi_completion_vllm,
    #   compute_entropy,
    # calculate_conditional_perplexity, calculate_conditional_perplexity_fast,
    # compute_diversity
    # calculate_conditional_perplexity_with_kv_cache
)
from utils import aws_s3_sync

if not hasattr(modeling_utils, "ALL_PARALLEL_STYLES") or modeling_utils.ALL_PARALLEL_STYLES is None:
    modeling_utils.ALL_PARALLEL_STYLES = ["tp", "none","colwise",'rowwise']

# model_name = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"


def main(seed, model_name, num_newline, log_file_name, save_dir_name, N):
    a = torch.randn(3, 3).cuda()
    print(a)

    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    vllm_model = LLM(
        model=model_name,
        enable_prefix_caching=True,
        max_model_len=40960
    )

    save_dir = f'./math_outputs'

    # model_solutions_dir = f"./model_solutions/{log_file_name}"
    model_solutions_dir = f"./required_files/{log_file_name}"
    model_solutions = []
    with open(model_solutions_dir, 'r') as f:
        for line in f:
            if line.strip():
                model_solutions.append(json.loads(line.strip()))


    all_answers = []

    sample_params_completion = SamplingParams(
        n=N,
        temperature=0.6,
        max_tokens=1000,
        top_p=0.95,
        top_k=-1,
        seed=seed
    )

    for qid, row in tqdm(
            # reversed(list(enumerate(model_solutions)))
            list(enumerate(model_solutions))
        ):
        # if qid < 380:
        #     continue
        model_response = (row['response'])
        model_response = model_response.split('</think>')[0]
        header = model_response.split("<think>")[0] + "<think>\n"
        if 'DeepSeek' in model_name:
            # reasoning_line = model_response.split("<think>")[1].split('\n')[1::2]
            reasoning_line = model_response.split("<think>")[1].split('\n\n')
            reasoning_line[0] = reasoning_line[0].replace('\n', '')
            reasoning_line[-1] = reasoning_line[-1].replace('\n', '')
        else:
            reasoning_line = model_response.split("<think>")[1].split('\n\n')
            reasoning_line[0] = reasoning_line[0][1:]
        answer_per_step = []
        for i in tqdm(range(len(reasoning_line) + 1)):
            partial_answer = (header + "\n\n".join(reasoning_line[:i + 1])) + "\n</think>" + "\n" * num_newline + 'Final Answer:'
            # print(partial_answer)
            early_stop_answers = get_multi_completion_vllm(
                vllm_model,
                partial_answer,
                # num_generations=16,
                # temperature=0.6,
                # max_tokens=1000,
                # seed=seed
                sample_params=sample_params_completion
            )
            answer_per_step.append(early_stop_answers)
        # all_answers.append(answer_per_step)


        local_save_dir = '/tmp'
        s3_save_dir_base = 's3://netflix-dataoven-prod-users/xiw/dpsk_math_eval/'
        file_name = f'answers_{qid}.json'
        folder_name = f'{save_dir_name}/seed_{args.seed}'
        # folder_name = f'dpsk_new_det_with_final_answer/seed_{args.seed}' # In this folder, we add "final answer" in the end of thinking

        save_dir = os.path.join(local_save_dir, folder_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        with open(os.path.join(save_dir, file_name), 'w') as f:
            json.dump(answer_per_step, f, indent=4)

        s3_save_dir_full = os.path.join(s3_save_dir_base, folder_name)

        print(save_dir)
        print(s3_save_dir_full)

        aws_s3_sync(
            save_dir,
            s3_save_dir_full,
            only_show_errors=False,
        )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--model', type=str, default='deepseek-ai/DeepSeek-R1-0528-Qwen3-8B')
    parser.add_argument('--num_newline', type=int, default=1)
    parser.add_argument('--log_file_name', type=str, default='dpsk_new_no_sample.json')
    parser.add_argument('--save_dir_name', type=str, default='dpsk_new_det_with_final_answer')
    parser.add_argument('--N', type=int, default=16)
    args = parser.parse_args()
    print(args)
    seed = args.seed
    model = args.model
    num_newline = args.num_newline
    log_file_name = args.log_file_name
    save_dir_name = args.save_dir_name
    N = args.N
    main(seed, model, num_newline, log_file_name, save_dir_name, N)
