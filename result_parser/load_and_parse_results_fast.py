import numpy as np
import matplotlib.pyplot as plt
import os
import boto3
import json
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from functools import partial
import multiprocessing as mp

from sal.utils.math import (
    find_majority_answer, extract_completion_answers,
    memoized_canonical_form, find_answer_with_largest_sum,
    strip_string
)

# print(repr(memoized_canonical_form('p - q', timeout_seconds=30)))

from tqdm import tqdm
from datasets import Dataset, load_dataset

def download_and_load_json(full_path, local_path):
    file_dir = full_path
    bucket_name, key = file_dir.replace('s3://', '').split('/', 1)
    s3 = boto3.client('s3')
    # Remove local_path if it exists
    try:
        os.remove(local_path)
    except OSError:
        pass
    s3.download_file(bucket_name, key, local_path)
    with open(local_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def download_seed_data(seed_qid_tuple):
    """Download data for a specific seed and qid"""
    seed, qid = seed_qid_tuple
    file_dir = f's3://netflix-dataoven-prod-users/xiw/dpsk_math_eval/dpsk_new_det_with_final_answer/seed_{seed}/answers_{qid}.json'
    local_path = f'/tmp/answers_{qid}_{seed}.json'
    return download_and_load_json(file_dir, local_path)

def process_answer_batch(args):
    """Process a batch of answers"""
    idx, row, gt_answer, qid = args
    extracted_answers = [
        memoized_canonical_form(a, timeout_seconds=60) 
        for a in extract_completion_answers({'completions': row})['preds']
    ]
    true_answer = memoized_canonical_form(gt_answer, timeout_seconds=60)
        # print(extracted_answers)
    acc = np.sum([extracted_answer == true_answer 
                  for extracted_answer in extracted_answers]) / len(extracted_answers)
    return idx, extracted_answers, acc 

def process_qid(qid, gt_answers, num_seeds=16, num_workers=8):
    """Process all data for a single question ID"""
    # Download all seed data in parallel
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        seed_qid_tuples = [(seed, qid) for seed in range(num_seeds)]
        all_answers = list(executor.map(download_seed_data, seed_qid_tuples))
    
    # Reshape data
    all_answers = np.array(all_answers)
    all_answers = all_answers.transpose(1, 0, 2).reshape(-1, 256)
    outputs = {}
    # Process answers in parallel
        # Prepare arguments
    process_args = [(idx, row, gt_answers[qid], qid) for idx, row in enumerate(all_answers)]
        
        # Submit all tasks
    futures = [process_answer_batch(arg) for arg in process_args]
        
        # Collect results with progress bar
        
    _all_answers = []
    accs = []
    for future in tqdm(futures, total=len(futures), desc=f"Processing QID {qid}"):
        idx, extracted_answers, acc = future.result()
        outputs[idx] = (
            extracted_answers, acc
        )
    

    # with ProcessPoolExecutor(max_workers=num_workers) as executor:
    #     # Prepare arguments
    #     process_args = [(idx, row, gt_answers[qid], qid) for idx, row in enumerate(all_answers)]
        
    #     # Submit all tasks
    #     futures = [executor.submit(process_answer_batch, arg) for arg in process_args]
        
    #     # Collect results with progress bar
        
    #     _all_answers = []
    #     accs = []
    #     for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing QID {qid}"):
    #         idx, extracted_answers, acc = future.result()
    #         outputs[idx] = (
    #             extracted_answers, acc
    #         )
    
    # Return sorted
    sorted_outputs = sorted(outputs.items(), key=lambda x: x[0])
    _all_answers = [output[1][0] for output in sorted_outputs]
    accs = [output[1][1] for output in sorted_outputs]
    return accs, _all_answers

def main():
    # Load dataset
    dataset_name = "HuggingFaceH4/MATH-500"
    dataset_split = "test"
    dataset = load_dataset(dataset_name, split=dataset_split)
    gt_answers = [row['answer'] for row in dataset]
    
    # Configuration
    start_qid = 1
    end_qid = 200
    num_workers = min(8, mp.cpu_count())  # Adjust based on your system
    tmp_dir = '/root/tts_and_entropy/outputs/cached_results_fast'
    
    # Ensure output directory exists
    os.makedirs(tmp_dir, exist_ok=True)
    
    # Process each QID
    for qid in range(start_qid, end_qid):
        print(f"\nProcessing QID {qid}")
        
        try:
            accs, _all_answers = process_qid(qid, gt_answers, num_workers=num_workers)
            
            # Save results
            np.save(os.path.join(tmp_dir, f'all_accs_{qid}.npy'), np.array(accs))
            np.save(os.path.join(tmp_dir, f'all_extracted_answers_{qid}.npy'), np.array(_all_answers))
            
        except Exception as e:
            print(f"Error processing QID {qid}: {e}")
            continue

if __name__ == "__main__":
    main()