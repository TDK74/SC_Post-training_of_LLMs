import re
import warnings
import pandas as pd
import torch

from datasets import Dataset, load_dataset
from helper import (generate_responses, load_model_and_tokenizer, test_model_with_questions)
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import GRPOConfig, GRPOTrainer


## ------------------------------------------------------ ##
warnings.filterwarnings('ignore')

## ------------------------------------------------------ ##
USE_GPU = False

SYSTEM_PROMPT = ("You are a helpful assistant that solves problems step-by-step. Always include "
                "the final numeric answer inside \\boxed{}.")

## ------------------------------------------------------ ##
def reward_func(completions, ground_truth, **kwargs):
    matches = [re.search(r"\\boxed\{(.*?)\}",
                        completion[0]['content']) for completion in completions]

    contents = [match.group(1) if match else "" for match in matches]

    return [1.0 if c == gt else 0.0 for c, gt in zip(contents, ground_truth)]

## ------------------------------------------------------ ##
sample_pred = [[{"role" : "assistant", "content" : r"...Calculating the answer. \boxed{72}"}]]
ground_truth = ["72"]
reward = reward_func(sample_pred, ground_truth)
print(f"Positive Sample Reward: {reward}")

## ------------------------------------------------------ ##
sample_pred = [[{"role" : "assistant", "content" : r"...Calculating the answer \boxed{71}"}]]
ground_truth = ["72"]
reward = reward_func(sample_pred, ground_truth)
print(f"Negative Sample Reward: {reward}")

## ------------------------------------------------------ ##
data_num = 5
eval_dataset = load_dataset("openai/gsm8k", "main")["test"].select(range(data_num))
sample_df = eval_dataset.to_pandas()
display(sample_df)

## ------------------------------------------------------ ##
def post_processing(example):
    match = re.search(r"####\s*(-?\d+)", example["answer"])

    example["ground_truth"] = match.group(1) if match else None

    example["prompt"] = [{"role" : "system", "content" : SYSTEM_PROMPT},
                        {"role" : "user", "content" : example["question"]}]

    return example


eval_dataset = eval_dataset.map(post_processing).remove_columns(["question", "answer"])

## ------------------------------------------------------ ##
sample_df = eval_dataset.select(range(5)).to_pandas()
display(sample_df)

## ------------------------------------------------------ ##
model, tokenizer = load_model_and_tokenizer("./models/Qwen/Qwen2.5-0.5B-Instruct", USE_GPU)

## ------------------------------------------------------ ##
all_preds = []
all_labels = []

for example in tqdm(eval_dataset):
    input_prompt = example["prompt"]
    ground_truth = example["ground_truth"]

    with torch.no_grad():
        response = generate_responses(model, tokenizer, full_message = input_prompt)

    all_preds.append([{"role" : "assistant", "content" : response}])
    all_labels.append(ground_truth)
    print(response)
    print("Ground truth: ", ground_truth)

rewards = reward_func(all_preds, all_labels)

accuracy = sum(rewards) / len(rewards)
print(f"Evaluation Accuracy: {accuracy:.2%}")

del model, tokenizer

## ------------------------------------------------------ ##
dataset = load_dataset("openai/gsm8k", "main")
train_dataset = dataset["train"]

train_dataset = train_dataset.map(post_processing)
train_dataset = train_dataset.remove_columns(["question", "answer"])

if not USE_GPU:
    train_dataset = train_dataset.select(range(10))

print(train_dataset[0])

## ------------------------------------------------------ ##
config = GRPOConfig(per_device_train_batch_size = 1,
                    gradient_accumulation_steps = 8,
                    num_generations = 4,
                    num_train_epochs = 1,
                    learning_rate = 5e-6,
                    logging_steps = 2,
                    no_cuda = not USE_GPU)

## ------------------------------------------------------ ##
model, tokenizer = load_model_and_tokenizer("./models/HuggingFaceTB/SmolLM2-135M-Instruct", USE_GPU)

grpo_trainer = GRPOTrainer(model = model,
                            args = config,
                            reward_funcs = reward_func,
                            train_dataset = train_dataset)

grpo_trainer.train()

## ------------------------------------------------------ ##
fully_trained_qwen = True

if fully_trained_qwen:
    model, tokenizer = load_model_and_tokenizer("./models/banghua/Qwen2.5-0.5B-GRPO", USE_GPU)

else:
    model = grpo_trainer.model

all_preds = []
all_labels = []

for example in tqdm(eval_dataset):
    input_prompt = example["prompt"]
    ground_truth = example["ground_truth"]

    with torch.no_grad():
        response = generate_responses(model, tokenizer, full_message = input_prompt)

    all_preds.append([{"role" : "assistant", "content" : response}])
    all_labels.append(ground_truth)
    print(response)
    print("Ground truth: ", ground_truth)

rewards = reward_func(all_preds, all_labels)

accuracy = sum(rewards) / len(rewards)
print(f"Evaluation Accuracy: {accuracy:.2%}")
