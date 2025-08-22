import numpy as np
import os
from tqdm import tqdm
import argparse

# Add random seed as argument
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--seed', type=int, default=1)
args = parser.parse_args()  

seed = args.seed

np.random.seed(args.seed)

MAX_RANGE = 500
# MAX_RANGE = 250

tmp_dir = '/root/tts_and_entropy/outputs/cached_results_dpsk_long_use_sample'
all_anss = []
all_counts = []
for id in tqdm(list(range(MAX_RANGE))):
    ANS = np.load(
        os.path.join(tmp_dir, f'all_extracted_answers_{id}.npy')
    )
    all_anss.append(ANS)
    COUNT = np.load(
        os.path.join(tmp_dir, f'all_extracted_answers_len_{id}.npy')
    )
    all_counts.append(COUNT)


subsampled_unique_counts_all = []
for n in [2, 4, 8, 16, 32]:
    subsampled_unique_counts = []
    for qid in tqdm(range(MAX_RANGE)):
        unique_counts = []
        num_tokens = []
        for row, token_count in zip(all_anss[qid], all_counts[qid]):
            idx = np.array([np.random.choice(len(row), size=n, replace=False) for _ in range(64)])
            _unique_counts = np.array([len(np.unique(_row)) for _row in row[idx]])
            _number_tokens = token_count[idx].sum()
            unique_counts.append(_unique_counts)
            num_tokens.append(_number_tokens)
        unique_counts = np.array(unique_counts)
        num_tokens = np.array(num_tokens)
        subsampled_unique_counts.append(np.array((unique_counts, num_tokens)))
    subsampled_unique_counts_all.append(subsampled_unique_counts)

save_dir = '/root/tts_and_entropy/ua_results_cache'

# Save subsampled_unique_counts_all as pkl
import pickle
# with open(os.path.join(save_dir, f'subsampled_unique_counts_all_{seed}.pkl'), 'wb') as f:
#     pickle.dump(subsampled_unique_counts_all, f)

with open(os.path.join(save_dir, f'subsampled_unique_counts_all_{seed}.pkl'), 'wb') as f:
    pickle.dump(subsampled_unique_counts_all, f)
