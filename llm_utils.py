model_name_mapper = {
    'dpsk_small': 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B',
    'dpsk_new': 'deepseek-ai/DeepSeek-R1-0528-Qwen3-8B',
    'qwen_new': 'Qwen/Qwen3-4B',
    'qwen_new_mid': 'Qwen/Qwen3-8B'
}


greedy_decode_config = {
    'do_sample': False,
}

deepseek_decode_config = {
    'do_sample': True,
    'temperature': 0.6,
    'top_p': 0.95,
}