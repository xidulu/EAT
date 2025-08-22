import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from transformers import AutoTokenizer

argparse = argparse.ArgumentParser()
argparse.add_argument('--folder_name', type=str)
argparse.add_argument('--save_name', type=str)
argparse.add_argument('--num_seed', type=int, default=8)
argparse.add_argument('--max_num', type=int, default=500)
argparse.add_argument('--min_num', type=int, default=-1)
args = argparse.parse_args()

folder_name = args.folder_name
save_name = args.save_name
num_seed = args.num_seed
max_num = args.max_num
min_num = args.min_num

model_name = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)


from sal.utils.math import (
    find_majority_answer, extract_completion_answers,
    memoized_canonical_form, find_answer_with_largest_sum,
    strip_string
)

from tqdm import tqdm
from datasets import Dataset, load_dataset

import boto3
import json
def download_and_load_json(full_path, local_path):
    file_dir = full_path
    bucket_name, key = file_dir.replace('s3://', '').split('/', 1)
    s3 = boto3.client('s3')
    # Remove local_path if it exists
    try:
        os.remove(local_path)
    except OSError:
        pass
    try:
        s3.download_file(bucket_name, key, local_path)
    except Exception as e:
        print(f"Error downloading file: {e}")
        print(key)

    # print(f"File downloaded to {local_path}")
    model_responses = []
    with open(local_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        return data

def eval_pass_at_k(candidates, answer, k):
    pass

dataset_name = "opencompass/AIME2025"
dataset_split = "test"
dataset = load_dataset(dataset_name, 'AIME2025-II', split=dataset_split)
gt_answers = [
    row['answer']
    for row in dataset
]

all_accs = []
all_extracted_answers = []
all_answer_len = []
ids = []

all_seeds = list(range(num_seed))
# all_seeds.remove(5)

num_seed = len(all_seeds)

for qid in range(max_num):
    if qid < min_num:
        continue
    all_answers = []
    for seed in all_seeds:
        # file_dir = f's3://netflix-dataoven-prod-users/xiw/dpsk_math_eval/dpsk_new_det_with_final_answer/seed_{seed}/answers_{qid}.json'
        file_dir = f's3://netflix-dataoven-prod-users/xiw/dpsk_math_eval/{folder_name}/seed_{seed}/answers_{qid}.json'
        F = download_and_load_json(file_dir, f'/tmp/answers_{save_name}.json')
        all_answers.append(F)

    all_answers = np.array(all_answers)
    all_answers = all_answers.transpose(1, 0, 2).reshape(-1, num_seed * 16)
    accs = []
    _all_answers = []
    _answer_lens = []
    for row in tqdm(all_answers):
        answer_len = [tokenizer(_row, return_tensors='pt').input_ids.shape[1] for _row in row]
        extracted_answers = [
            memoized_canonical_form(a, timeout_seconds=10) for a in extract_completion_answers(
            {'completions': row},
        )['preds']]

        _all_answers.append(extracted_answers)
        _answer_lens.append(answer_len)
        true_answer = memoized_canonical_form(gt_answers[qid])
        accs.append(
            np.sum([extracted_answer == true_answer
            for extracted_answer in extracted_answers]) / len(extracted_answers)
        )
    ids.append(qid)
    all_accs.append(accs)
    all_extracted_answers.append(_all_answers)
    all_answer_len.append(_answer_lens)


    tmp_dir = f'/root/tts_and_entropy/outputs/cached_results{save_name}'

    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    np.save(tmp_dir + f'/all_accs_{qid}.npy', np.array(accs))

    np.save(tmp_dir + f'/all_extracted_answers_{qid}.npy', np.array(_all_answers))
    np.save(tmp_dir + f'/all_extracted_answers_len_{qid}.npy', np.array(_answer_lens))
