# EAT

Code for EAT: Entropy After `</Think>` for reasoning model early exiting. (PAPER LINK)

The full dataset can be found at [text](https://huggingface.co/datasets/xidulu/eat_rollouts)


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
# Take in a reasoning chain file and generate answer rollouts for each reasoning step (split by \n\n)
# by appending "</think> Final answer:" and let the model generates till <EoS>.
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

the script above will parse all the json files in the folder and compute the Pass@1 (Avg@128) at each reasoning step and save the results in the save_dir.

## Utils

**dataset_utils.py**

**compute_entropy_ngram.py**

**early_stopping_utils.py**

**generation_utils.py**

**llm_utils.py**


# To reproduce the results
