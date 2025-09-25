from datasets import Dataset, load_dataset

system_prompt = r"Solve the following math problem efficiently and clearly. Please reason step by step, and put your final answer within \boxed{}."

    
def load_math500():
    dataset_name = "HuggingFaceH4/MATH-500"
    dataset_split = "test"
    dataset = load_dataset(dataset_name, split=dataset_split)
    convs = [
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        for prompt in dataset["problem"]
    ]
    return convs


def load_aime2025():
    dataset_name = "opencompass/AIME2025"
    dataset_split = "test"
    dataset = load_dataset(dataset_name, 'AIME2025-I', split=dataset_split)
    convs = [
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        for prompt in dataset["question"]
    ]
    return convs


def load_aime2025_2():
    dataset_name = "opencompass/AIME2025"
    dataset_split = "test"
    dataset = load_dataset(dataset_name, 'AIME2025-II', split=dataset_split)
    convs = [
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        for prompt in dataset["question"]
    ]
    return convs


def load_gpqa():
    dataset_name = "Idavidrein/gpqa"
    dataset_split = "train"
    dataset = load_dataset(dataset_name, 'gpqa_diamond', split=dataset_split)
    convs = [
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        for prompt in dataset["Pre-Revision Question"]
    ]
    return convs



def load_gpqa_mc():

    def process_row(row):
        # Turn a row into a multiple choice QA format
        Q = row['Question']
        options = row['options']
        options = [f"{chr(65 + i)}. {opt}" for i, opt in enumerate(options)]
        # Compose a string
        options_str = "\n".join(options)
        return f"Question:\n{Q}\nChoices:\n{options_str}"

    system_prompt = r"Solve the following multiple choice problem efficiently and clearly. Please reason step by step, and return one single option as the answer, put your answer within \boxed{}."
    dataset_name = "jeggers/gpqa_formatted"
    dataset_split = "train"
    dataset = load_dataset(dataset_name, 'diamond', split=dataset_split)
    convs = [
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": process_row(row)},
        ]
        for row in dataset
    ]
    return convs



def build_deepseekcoder_instruction(languge: str, question: str):
    return '''
Please continue to complete the function. You are not allowed to modify the given code and do the completion only. Please return all completed function in a codeblock. Here is the given code to do completion:
```{}
{}
```
'''.strip().format(languge.lower(), question.strip())

def load_humaneval():
    dataset = load_dataset("openai/openai_humaneval", split="test")
    convs = [
        [
            {"role": "user", "content": build_deepseekcoder_instruction('Python', prompt)},
        ]
        for prompt in dataset["prompt"]
    ]
    return convs


dataset_mapping = {
    "math500": load_math500,
    "aime2025": load_aime2025,
    "aime2025_2": load_aime2025_2,
    "gpqa": load_gpqa,
    'human_eval': load_humaneval,
    'gpqa_mc': load_gpqa_mc,
}