from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import modeling_utils, DynamicCache
from tqdm import tqdm
from copy import deepcopy
import json
import time
import argparse

# Add arg
parser = argparse.ArgumentParser(description="Estimate time for model inference")
parser.add_argument("--seed", type=int, default=1)
args = parser.parse_args()

seed = args.seed
torch.manual_seed(seed)
np.random.seed(seed)



if not hasattr(modeling_utils, "ALL_PARALLEL_STYLES") or modeling_utils.ALL_PARALLEL_STYLES is None:
    modeling_utils.ALL_PARALLEL_STYLES = ["tp", "none","colwise",'rowwise']


device = f"cuda:{seed}"
model_name = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.to(device)

think_token = "</think>"
think_token_id = tokenizer.encode(think_token, add_special_tokens=False)[0]


model_solutions_dir = '/root/tts_and_entropy/model_solutions/dpsk_new_with_sample_long.json'
model_solutions_dpsk_long_with_sample = []
with open(model_solutions_dir, 'r') as f:
    for line in f:
        if line.strip():
            model_solutions_dpsk_long_with_sample.append(json.loads(line.strip()))

R = model_solutions_dpsk_long_with_sample[188]['response']
R = R.split('</think>')[0]
header = R.split("<think>")[0] + '<think>\n'
reasoning_line = R.split("<think>")[1].split('\n\n')
reasoning_line[0] = reasoning_line[0][1:]  


INTERVAL = 1
with torch.no_grad():
    next_token_entropy = []
    entropies = []
    bi_gram_entropies = []
    cache = DynamicCache()
    for i in range(len(reasoning_line) + 1)[::INTERVAL]:
        partial_answer = (header + "\n\n".join(reasoning_line[:i]))
        inputs_big = tokenizer(partial_answer, return_tensors="pt")['input_ids'].to(device)
        if cache is not None:
            inputs = inputs_big[:, cache.get_seq_length():]
        model_outputs = model(
            inputs,
            output_hidden_states=False,
            return_dict=True,
            cache=True,
            past_key_values=cache if cache is not None else None,
        )
        cache = model_outputs.past_key_values


CACHE_ROLLOUT = deepcopy(cache)

lengths = [
    # 8192, 4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2
    8000, 6000, 4000, 2000, 1000, 500
]
all_times_rollouts = []
answer_lengths = []

with torch.no_grad():
    for i in range(len(lengths)):
        inputs = tokenizer("\n</think>\nFinal answer:", return_tensors="pt")['input_ids'].to(device)
        _time = []
        _answer_length = []
        for _ in range(16):
            CACHE_ROLLOUT.crop(lengths[i])
            st = time.time()
            rollouts = model.generate(
                torch.concat([
                    inputs_big[:, :lengths[i]],
                    inputs
                ], 1),
                max_new_tokens=2000,
                do_sample=True,
                temperature=0.6,
                use_cache=True,
                past_key_values=CACHE_ROLLOUT,
                output_scores=False,
                return_dict_in_generate=True,
                top_k=None,
                top_p=None,
            )
            et = time.time()
            _answer_length.append(rollouts.sequences.shape[1] - lengths[i])
            _time.append(et - st)
        all_times_rollouts.append(_time)
        answer_lengths.append(_answer_length)

all_times_rollouts = np.array(all_times_rollouts)
np.save(f"/root/tts_and_entropy/runtime_results/rollout_times_{seed}.npy", all_times_rollouts)
answer_lengths = np.array(answer_lengths)
np.save(f"/root/tts_and_entropy/runtime_results/answer_lengths_{seed}.npy", answer_lengths)