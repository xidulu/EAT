# EAT

Code for EAT: Entropy After `</Think>` for reasoning model early exiting. (PAPER LINK)

**Status**:  Not open sourced yet, will make the repo public once the paper is on arxiv and the code is little bit more organized.

# What we have here?

- **./notebooks** All code used for creating the figures 

- **./requirment_files** The dependency files

- **./result_parser** For extracting the answers from the rollouts

- **compute_entropy_ngram.py** The file for computing EAT given a CoT file

- **dataset_utils.py** Data loader for loading the reasoning datasets

- **early_stopping_utils.py**

- **estimate_time.py** Estimate the runtime of generating rollouts 

- **generate_math_cot.py** Generate reasoning chain for math problems

- **generate_code_cot.py** Generate reasoning chain for coding problems

- **generation_utils.py** Helper function for generating answer rollouts (wrapper of vLLM), and the function for computing entropy with KV support.

- **get_ua_results** Code for estimating number of unique answers in multiple rollouts.

- **llm_utils.py** Some configuration files

- **generate_answer_rollouts.py** Take in a CoT file, generate answer string at each step