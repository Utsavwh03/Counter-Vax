import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import torch
from transformers import pipeline
import peft ,trl

# 'load_dataset' is a function from the 'datasets' library by Hugging Face which allows you to load a dataset.
from datasets import load_dataset

from peft import LoraConfig, prepare_model_for_kbit_training, TaskType, PeftModel

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    set_seed
)

# # 'SFTTrainer' is a class from the 'trl' library that provides a trainer for soft fine-tuning.
from trl import SFTTrainer

pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.bfloat16, device_map="auto")

# # We use the tokenizer's chat template to format each message - see https://huggingface.co/docs/transformers/main/en/chat_templating
# messages = [
#     {
#         "role": "system",
#         "content": "You are a friendly chatbot who generates a brief counter-arguments to any statement",
#     },
#     {"role": "user", "content": "Vaccine are bad for health "},
# ]
# prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
# outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
# print(outputs[0]["generated_text"])
tokenizer=pipe.tokenizer
def create_message_column(row):
        # Initialize an empty list to store the messages.
    messages = []
    
    # Create a 'user' message dictionary with 'content' and 'role' keys.
#     print(row['text'])
    user = {
        "content": f"Generate Counter Argument for the tweet:\n Tweet: {row['text']}",
        "role": "user"
    }
    
    # Append the 'user' message to the 'messages' list.
    messages.append(user)
    
    # Create an 'assistant' message dictionary with 'content' and 'role' keys.
    assistant = {
        "content": f"{row['counter_argument']}",
        "role": "assistant"
    }
    
    # Append the 'assistant' message to the 'messages' list.
    messages.append(assistant)
    
    # Return a dictionary with a 'messages' key and the 'messages' list as its value.
    return {"messages": messages}

def format_dataset_chatml(row):
    return {"text": tokenizer.apply_chat_template(row["messages"], add_generation_prompt=False, tokenize=False)}


from datasets import load_dataset
train_data= load_dataset('csv', data_files='/home/sohampoddar/HDD2/utsav/Data/train_CA_with_label_desc.csv',split='train')
val_data= load_dataset('csv', data_files='/home/sohampoddar/HDD2/utsav/Data/val_CA_without_labels.csv',split='train')


dataset_chatml = train_data.map(create_message_column)

dataset_chatml = dataset_chatml.map(format_dataset_chatml)

val_dataset_chatml=val_data.map(create_message_column)
val_dataset_chatml=val_dataset_chatml.map(format_dataset_chatml)



model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_id)


if torch.cuda.is_bf16_supported():
  compute_dtype = torch.bfloat16
  attn_implementation = 'flash_attention_2'
# If bfloat16 is not supported, 'compute_dtype' is set to 'torch.float16' and 'attn_implementation' is set to 'sdpa'.
else:
  compute_dtype = torch.float16
  attn_implementation = 'sdpa'

# # This line of code is used to print the value of 'attn_implementation', which indicates the chosen attention implementation.
# print(attn_implementation)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # Use EOS token as padding token if none is defined

# # Set the pad_token_id
tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
# Set padding side if needed
tokenizer.padding_side = 'right'
# set cuda available device
if torch.cuda.is_available():
    device = torch.device("cuda")  # 'cuda' will point to the visible GPU (e.g., GPU 0 as set by CUDA_VISIBLE_DEVICES)
    print("Using GPU:", torch.cuda.get_device_name(device))
else:
    device = torch.device("cpu")
    print("Using CPU")

use_4bit = True
# 'bnb_4bit_compute_dtype' is the data type that should be used for computations with the 4-bit base model. In this case, it is set to 'bfloat16'.
bnb_4bit_compute_dtype = "bfloat16"

# 'bnb_4bit_quant_type' is the type of quantization that should be used for the 4-bit base model. In this case, it is set to 'nf4'.
bnb_4bit_quant_type = "nf4"

# 'use_double_quant' is a boolean that controls whether nested quantization should be used for the 4-bit base model.
use_double_quant = True
bnb_config= BitsAndBytesConfig(
      load_in_4bit=True,
      bnb_4bit_use_double_quant=True,
      bnb_4bit_quant_type="nf4",
      bnb_4bit_compute_dtype=torch.bfloat16
  )

model = AutoModelForCausalLM.from_pretrained(
          model_id, torch_dtype=compute_dtype, trust_remote_code=True, quantization_config=bnb_config, device_map="auto",
#           attn_implementation=attn_implementation
)
model.gradient_checkpointing_enable()  # reduce number of stored activations
model = prepare_model_for_kbit_training(model)
target_modules= ['k_proj', 'q_proj', 'v_proj', 'o_proj', "gate_proj", "down_proj", "up_proj"]
peft_config = LoraConfig(
        r=4,
        lora_alpha=32,
        lora_dropout=0.1,
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
)
model = get_peft_model(model, peft_config)
args = TrainingArguments(
        output_dir="./tiny-llama-1.1b-chat-v1.0",
        evaluation_strategy="epoch",
        do_eval=True,
        optim="adamw_torch",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        per_device_eval_batch_size=2,
        log_level="debug",
        save_strategy="epoch",
        learning_rate=1e-4,
        logging_dir=None,  # Directory for storing logs
        logging_steps=100,     # Log after every 100 steps
        logging_strategy="steps",  # Log by steps instead of epochs
        fp16=True,
#         bf16 = torch.cuda.is_bf16_supported(),
        eval_steps=100,
        num_train_epochs=2,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",

        seed=42,
)

import os
from rouge_score import rouge_scorer

trainer = SFTTrainer(
        model=model,
        train_dataset=dataset_chatml,
        eval_dataset=val_dataset_chatml,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=512,
#         compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        args=args,
)
trainer.train()

trainer.save_model()


#------Evaluation-------

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoConfig
model_path = "/home/sohampoddar/HDD2/utsav/tiny-llama-1.1b-chat-v1.0"
tokenizer_path = "/home/sohampoddar/HDD2/utsav/tiny-llama-1.1b-chat-v1.0"
# connect to device
if torch.cuda.is_available():
    device = torch.device("cuda")  # 'cuda' will point to the visible GPU (e.g., GPU 0 as set by CUDA_VISIBLE_DEVICES)
    print("Using GPU:", torch.cuda.get_device_name(device))
else:
    device = torch.device("cpu")
    print("Using CPU")

model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=compute_dtype, trust_remote_code=True, quantization_config=bnb_config, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

import evaluate
from evaluate import load  # Correct import for metrics
metric = evaluate.load("rouge")

from transformers import pipeline
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

val_dataset_chatml=val_data.map(create_message_column)
val_dataset_chatml=val_dataset_chatml.map(format_dataset_chatml)
dataset_chatml = val_dataset_chatml.train_test_split(test_size=0.5)
dataset_chatml

def test_inference(prompt):
    prompt = pipe.tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    outputs = pipe(prompt, max_new_tokens=100, do_sample=True, num_beams=1, temperature=0.7, top_k=50, top_p=0.9,repetition_penalty=1.2, max_time= 50)
    # print(outputs[0]['generated_text'])
    return outputs[0]['generated_text'][len(prompt):].strip()

print(dataset_chatml['test']['text'][1])


num_samples=50
prompts = [pipe.tokenizer.apply_chat_template([{"role": "user", "content": dataset_chatml['test'][i+50]['messages'][0]['content']}], tokenize=False, add_generation_prompt=True)
                                              for i in range(num_samples)]
outputs = pipe(prompts, batch_size=1, max_new_tokens=100, do_sample=True, num_beams=1, temperature=0.7, top_k=50, top_p=0.9,repetition_penalty=1.2,
                   max_time= 180)
# print(outputs)
preds = []
for i in range(len(outputs)):
    generated_text = outputs[i][0]['generated_text']
  
    response = generated_text[len(prompts[i]):].split()
    # print(response)
    if len(response) > 1:
        # Extract the counter argument
        pred = " ".join(response)
    else:
        pred = ""  # Handle case with no valid split
    preds.append(pred)

    # Print prediction and corresponding reference
    print(f"Prediction {i+50+ 1}: {pred}")
    print(f"Reference {i+50 + 1}: {dataset_chatml['test'][i+50]['counter_argument']}")
    print("---")  # Separator for clarity
references= [dataset_chatml['test'][i+50]['counter_argument'] for i in range(len(outputs))]
metric.add_batch(predictions=preds, references=references)

metric.add_batch(predictions=preds, references=references)
result = metric.compute(use_stemmer=True)

import numpy as np

# Assuming result contains your ROUGE scores
rouge1_mean = np.mean(result['rouge1']) * 100
rouge2_mean = np.mean(result['rouge2']) * 100
rougeL_mean = np.mean(result['rougeL']) * 100
rougeLsum_mean = np.mean(result['rougeLsum']) * 100

print("Rouge 1 Mean: ", rouge1_mean)
print("Rouge 2 Mean: ", rouge2_mean)
print("Rouge L Mean: ", rougeL_mean)
print("Rouge Lsum Mean: ", rougeLsum_mean)


from bert_score import score
P, R, F1 = score(preds, references, lang="en", verbose=True)

# Display BERTScore results
print("BERTScore Precision:", P.mean().item())
print("BERTScore Recall:", R.mean().item())
print("BERTScore F1:", F1.mean().item())