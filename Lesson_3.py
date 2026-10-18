import warnings
import pandas as pd
import torch

from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer
from helper import display_dataset


## ------------------------------------------------------ ##
warnings.filterwarnings('ignore')

## ------------------------------------------------------ ##
def generate_responses(model, tokenizer, user_message, system_message= None, max_new_tokens= 100):
    messages = []

    if system_message:
        messages.append({"role" : "system", "content" : system_message})

    messages.append({"role" : "user", "content" : user_message})

    prompt = tokenizer.apply_chat_template(messages,
                                            tokenize= False,
                                            add_generation_prompt= True,
                                            enable_thinking= False, )

    inputs = tokenizer(prompt, return_tensors= "pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(**inputs,
                                max_new_tokens= max_new_tokens,
                                do_sample= False,
                                pad_token_id= tokenizer.eos_token_id,
                                eos_token_id= tokenizer.eos_token_id, )

    input_len = inputs["input_ids"].shape[1]
    generated_ids = outputs[0][input_len : ]
    response = tokenizer.decode(generated_ids, skip_special_tokens= True).strip()

    return response

## ------------------------------------------------------ ##
def test_model_with_questions(model, tokenizer, questions, system_message= None,
                            title= "Model Output"):
    print(f"\n=== {title} ===")

    for i, question in enumerate(questions, 1):
        response = generate_responses(model, tokenizer, question, system_message)
        print(f"\nModel Input {i}:\n{question}\nModel Output {i}:\n{response}\n")

## ------------------------------------------------------ ##
def load_model_and_tokenizer(model_name, use_gpu= False):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    if use_gpu:
        model.to("cuda")

    if not tokenizer.chat_template:
        tokenizer.chat_template = """
                                    {% for message in messages %}
                                    {% if message['role'] == 'system' %}
                                    System: {{ message['content'] }}\n
                                    {% elif message['role'] == 'user' %}
                                    User: {{ message['content'] }}\n
                                    {% elif message['role'] == 'assistant' %}
                                    Assistant: {{ message['content'] }} <|endoftext|>
                                    {% endif %}
                                    {% endfor %}
                                """

    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

## ------------------------------------------------------ ##
USE_GPU = False

questions = ["Give me an 1-sentence introduction of LLM.",
            "Calculate 1+1-1",
            "What's the difference between thread and process?"]

## ------------------------------------------------------ ##
model, tokenizer = load_model_and_tokenizer("./models/Qwen/Qwen3-0.6B-Base", USE_GPU)

test_model_with_questions(model, tokenizer, questions, title= "Base Model (Before SFT) Output")

del model, tokenizer

## ------------------------------------------------------ ##
model, tokenizer = load_model_and_tokenizer("./models/banghua/Qwen3-0.6B-SFT", USE_GPU)

test_model_with_questions(model, tokenizer, questions, title= "Base Model (After SFT) Output")

del model, tokenizer

## ------------------------------------------------------ ##
model_name = "./models/HuggingFaceTB/SmolLM2-135M"
model, tokenizer = load_model_and_tokenizer(model_name, USE_GPU)

## ------------------------------------------------------ ##
train_dataset = load_dataset("banghua/DL-SFT-Dataset")["train"]

if not USE_GPU:
    train_dataset = train_dataset.select(range(100))

display_dataset(train_dataset)

## ------------------------------------------------------ ##
sft_config = SFTConfig(learning_rate= 8e-5,
                        num_train_epochs= 1,
                        per_device_train_batch_size= 1,
                        gradient_accumulation_steps= 8,
                        gradient_checkpointing= False,
                        logging_steps= 2, )

## ------------------------------------------------------ ##
sft_trainer = SFTTrainer(model= model,
                        args= sft_config,
                        train_dataset= train_dataset,
                        processing_class= tokenizer, )
sft_trainer.train()

## ------------------------------------------------------ ##
if not USE_GPU:
    sft_trainer.model.to("cpu")

test_model_with_questions(sft_trainer.model,
                        tokenizer,
                        questions,
                        title= "Base Model (After SFT) Output")
