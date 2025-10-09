# EAT

Code for EAT: Entropy After `</Think>` for reasoning model early exiting. (https://arxiv.org/abs/2509.26522)

```
@article{wang2025entropy,
  title={Entropy After $\langle \texttt{/Think}\rangle $ for reasoning model early exiting},
  author={Wang, Xi and McInerney, James and Wang, Lequn and Kallus, Nathan},
  journal={arXiv preprint arXiv:2509.26522},
  year={2025}
}
```

The full dataset can be [found here](https://huggingface.co/datasets/xidulu/eat_rollouts)


# To setup environment
```
uv venv eat
uv pip install -r ./requirement_files/requirments_latest.txt
source eat/bin/activate
```

# Files

## Rollout Generation

**generate_math_cot.py**:
```bash
# Generate full reasoning chain for all questions in a dataset
# Example:
python generate_math_cot.py --model dpsk_new \
      --dataset_name math500 --save_dir ./reasoning_chains/
```

**generate_answer_rollouts.py**:
```bash
# Take in a reasoning chain file and generate answer rollouts for each
# reasoning step (split by \n\n) by appending "</think> Final answer:"
# and let the model generates till <EoS>.
# By default each seed will generate 16 rollouts per reasoning step.
# Example of generating 128 rollouts per step:
for seed in {0..7}
do
python generate_answer_rollouts.py  --seed $seed \
--log_file_name DeepSeek-R1-0528-Qwen3-8B_math500.json \
--save_dir_name YOUR_SAVE_DIR \
done
```

## Evaluate Pass@1 (Avg@128)
We are interested in knowing the correctness of each rollout generated at each reasoning step.

```bash
PYTHONPATH=. python result_parser/load_and_parse_results_math500.py \
--folder_name ./rollouts/DeepSeek-R1-0528-Qwen3-8B_math500 \
--save_dir ./cached_results_DeepSeek-R1-0528-Qwen3-8B_math500_math500 \
--max_num 200 --num_seed 4 --min_num 0
```
the script above will parse all the json files in the folder and compute the Pass@1 (Avg@128)
at each reasoning step and save the results in the save_dir.

## Compute EAT
Given a reasoning trajectory, we compute the Entropy After `</think>` in a posthoc way,
i.e. instead of actually computing it online, we first generate a long reasoning chain,
then we compute the signal at different positions. 
In particular, we split the reasoning process at `\n\n`.

The script `compute_entropy_ngram.py` is for this purpose,
```bash
python compute_entropy_ngram.py \
    --model_solution_dir DeepSeek-R1-0528-Qwen3-8B_math500.json\
    --eval_model dpsk_new \
    --save_dir ./entropy_log \
    --save_name DeepSeek-R1-0528-Qwen3-8B_math500WithFinalAns \
    --gpu 0 \
    --include_final_answer_str \
```

The `eval_model` flag specifies which model to use for computing entropy,
`--include_final_answer_str` specifies whether a `'Final answer: '` string is included when computing
the entropy.

## Early stopping with EMA

The helper function for early stopping with EMA (in a post-hoc way) is in `early_stopping_utils.py`.
The main utility function is `choose_early_stop_point`, which takes in a 1D numpy array of EAT values
and returns the early stopping point (index) based on the EMA strategy.

```python
def choose_early_stop_point(entropy, timescale, threshold, min_distance=25, ema_0=0.0, min_value_threshold=None, normalize=False):
    ema_variance = exponential_moving_variance(entropy, timescale, ema_0, normalize=normalize)
    if min_distance > len(entropy):
        return -1
    if min_value_threshold is not None:
        min_value_pos = np.argmax(entropy < min_value_threshold)
        min_distance = max(min_value_pos, min_distance)
    exit_idx = get_early_stop_point_posthoc(ema_variance, threshold, min_distance, normalize=normalize)
    return exit_idx if exit_idx is not None else -1
```

The key parameters are:
- `timescale` (\alpha): the timescale for EMA
- `threshold` (\delta): the threshold for EMA variance to trigger early stopping
- `min_distance`: the minimum distance from the start to consider early stopping, typically set as 4 / \alpha for EMA warming up.

The function returns -1 if no early stopping point is found.


## Utils

**dataset_utils.py**

**compute_entropy_ngram.py**

**early_stopping_utils.py**

**generation_utils.py**

**llm_utils.py**


# To reproduce the results

TODO
