import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# release the gpu 
import gc
import torch
gc.collect()
torch.cuda.empty_cache()
import argparse
import numpy as np

# import all the necessary libraries
import pandas as pd
import torch
from datasets import Dataset
from peft import get_peft_model, LoraConfig, TaskType
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, default_data_collator

# LABEL DESCRIPTIONS
labels_map = {}
labels_map['religious'] = "religious beliefs and their influence on views about vaccines"
labels_map['political'] = "the political factors that affect perceptions of vaccine use"
labels_map['ingredients'] = "concerns about the ingredients and chemical components in vaccines"
labels_map['unnecessary'] = "the importance and necessity of getting vaccinated to prevent diseases"
labels_map['conspiracy'] = "conspiracy theories suggesting hidden motives behind vaccination efforts"
labels_map['mandatory'] = "the debate over personal choice versus mandates in vaccination policies"
labels_map['ineffective'] = "evidence and reasons that support the effectiveness of vaccines"
labels_map['side-effect'] = "potential side effects and adverse reactions associated with vaccines"
labels_map['pharma'] = "the role of pharmaceutical companies and concerns about profit motives"
labels_map['rushed'] = "claims that vaccines were approved or developed without sufficient testing"
labels_map['country'] = "national biases and objections to vaccines produced by specific countries"

# load the data
class Finetune_Model:
    def __init__(self, model_name, dataset_path, num_epochs, learning_rate, weight_decay, save_dir):
        self.model_name = model_name
        self.dataset_path = dataset_path
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.save_dir = save_dir

    def load_data(self):
        self.train_data = pd.read_csv(self.dataset_path)
        self.test_data = pd.read_csv(self.dataset_path)

    def load_model(self):
        # Use AutoModelForCausalLM to support both Llama and Gemma models
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        

    def create_message_column_with_labels(self, row):
        # Initialize an empty list to store the messages.
        messages = []
        prompt = f"Generate Counter Argument for the anti-vaccine tweet:\n Tweet: {row['text']}\n Talk About "

        lab = row['labels'].split()
        if isinstance(lab, list):
            mapped_labels = " and ".join([labels_map.get(l, "") for l in lab])
        else:
            mapped_labels = labels_map.get(lab, "")
        prompt += mapped_labels
        prompt += " ##Output: "
        
        user = {
            "content": prompt,
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

    def create_message_column_without_labels(self, row):
        messages = []
        
        prompt = f"Generate Counter Argument for the anti-vaccine tweet:\n Tweet: {row['text']}\n ##Output: "
        user = {
            "content": prompt,
            "role": "user"
        }
        messages.append(user)
        assistant = {
            "content": f"{row['counter_argument']}",
            "role": "assistant"
        }
        messages.append(assistant)
        return {"messages": messages}


    def format_dataset_chatml(self, row):
        # Apply chat template to create the text string
        # SFTTrainer will tokenize this automatically based on dataset_text_field
        # When assistant_only_loss=True, TRL processes the text and extracts labels internally
        text = self.tokenizer.apply_chat_template(
            row["messages"], 
            add_generation_prompt=False, 
            tokenize=False
        )
        # Return text field that SFTTrainer will process
        # Do NOT add labels here - let SFTTrainer handle tokenization and label creation
        return {"text": text}

    def setup_model_config(self):
        """Setup model configuration with quantization and attention implementation"""
        import torch
        from transformers import BitsAndBytesConfig, AutoModelForCausalLM
        from peft import prepare_model_for_kbit_training, LoraConfig, TaskType
        
        # Check if bfloat16 is supported
        if torch.cuda.is_bf16_supported():
            compute_dtype = torch.bfloat16
            attn_implementation = 'flash_attention_2'
        else:
            compute_dtype = torch.float16
            attn_implementation = 'sdpa'
        
        print(f"Using attention implementation: {attn_implementation}")
        
        # Setup tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)
        self.tokenizer.padding_side = 'right'
        
        # Update chat template to support assistant_only_loss
        # TRL requires {% generation %} markers to identify assistant tokens for loss calculation
        # Create a template compatible with Llama-3.2 format that includes generation markers
        llama_template_with_generation = """{%- for message in messages %}
            {%- if message['role'] == 'system' %}
            <|begin_of_text|><|start_header_id|>system<|end_header_id|>

            {{ message['content'] }}<|eot_id|>
            {%- elif message['role'] == 'user' %}
            <|start_header_id|>user<|end_header_id|>

            {{ message['content'] }}<|eot_id|>
            {%- elif message['role'] == 'assistant' %}
            <|start_header_id|>assistant<|end_header_id|>

            {% generation %}{{ message['content'] }}{% endgeneration %}<|eot_id|>
            {%- endif %}
            {%- endfor %}"""
        gemma_template_with_generation = """<bos>{%- for message in messages %}
{%- if message['role'] == 'system' %}
{{ message['content'] }}
{%- elif message['role'] == 'user' %}
<start_of_turn>user
{{ message['content'] }}<end_of_turn>
{%- elif message['role'] == 'assistant' %}
<start_of_turn>model
{% generation %}{{ message['content'] }}{% endgeneration %}<end_of_turn>
{%- endif %}
{%- endfor %}"""
        phi3_template_with_generation = """{%- for message in messages %}
{%- if message['role'] == 'system' %}
<|system|>
{{ message['content'] }}<|end|>
{%- elif message['role'] == 'user' %}
<|user|>
{{ message['content'] }}<|end|>
{%- elif message['role'] == 'assistant' %}
<|assistant|>
{% generation %}{{ message['content'] }}{% endgeneration %}<|end|>
{%- endif %}
{%- endfor %}"""
        if self.model_name == "google/gemma-2b-it":
            self.tokenizer.chat_template = gemma_template_with_generation
        elif self.model_name == "meta-llama/Llama-3.2-3B-Instruct":
            self.tokenizer.chat_template = llama_template_with_generation
        elif self.model_name in ["microsoft/Phi-3-mini-4k-instruct", "microsoft/phi-3-mini-4k-instruct", "microsoft/Phi-3-mini-128k-instruct"]:
            self.tokenizer.chat_template = phi3_template_with_generation
        else:
            raise ValueError(f"Model {self.model_name} not supported")
        print("Updated chat template to include {% generation %} markers for assistant_only_loss support")
        
        # Setup device
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("Using GPU:", torch.cuda.get_device_name(device))
        else:
            device = torch.device("cpu")
            print("Using CPU")
        
        # Setup quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        # Load model with quantization
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, 
            torch_dtype=compute_dtype, 
            trust_remote_code=True, 
            quantization_config=bnb_config, 
            device_map="auto"
        )
        
        # Enable gradient checkpointing and prepare for training
        self.model.gradient_checkpointing_enable()
        self.model = prepare_model_for_kbit_training(self.model)
        
        # Setup LoRA config
        # Phi-3 uses similar architecture to Llama, but let's use a more comprehensive set
        if "phi" in self.model_name.lower() or "Phi" in self.model_name:
            # Phi-3 target modules (similar to Llama architecture)
            # Try common modules - adjust if needed
            target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
        else:
            # Llama/Gemma target modules
            target_modules = ['k_proj', 'q_proj', 'v_proj', 'o_proj', "gate_proj", "down_proj", "up_proj"]
        
        self.peft_config = LoraConfig(
            r=4,
            lora_alpha=32,
            lora_dropout=0.1,
            task_type=TaskType.CAUSAL_LM,
            target_modules=target_modules,
        )
        
        return compute_dtype, attn_implementation

    def setup_training_args(self, compute_dtype):
        """Setup training arguments using SFTConfig"""
        from trl import SFTConfig
        
        args = SFTConfig(
            output_dir=self.save_dir,
            eval_strategy="epoch",  # Changed from evaluation_strategy to eval_strategy
            do_eval=True,
            optim="adamw_torch",
            per_device_train_batch_size=2,
            gradient_accumulation_steps=2,
            per_device_eval_batch_size=2,
            log_level="debug",
            save_strategy="epoch",
            learning_rate=self.learning_rate,
            logging_dir=None,
            logging_steps=100,
            logging_strategy="steps",
            bf16=True,  # Changed from fp16 to bf16 for Gemma stability (prevents NaN)
            eval_steps=100,
            num_train_epochs=self.num_epochs,
            warmup_ratio=0.1,
            lr_scheduler_type="linear",
            seed=42,
            completion_only_loss=True,    # <-- use this for non-conversational data (automatically masks non-completion tokens)
            dataset_text_field="text",  # Name of the text column in the dataset
            max_length=512,  # Maximum sequence length
            max_grad_norm=1.0,  # Gradient clipping to prevent NaN values
            warmup_steps=100,  # Warmup steps for stable training
        )
        return args

    def prepare_datasets(self):
        """Prepare training and validation datasets"""
        from datasets import load_dataset
        
        # Load datasets
        train_data = load_dataset('csv', data_files=self.dataset_path, split='train')
        val_data = load_dataset('csv', data_files=self.dataset_path.replace('train', 'val'), split='train')
        
        # Process datasets
        train_dataset_chatml = train_data.map(self.create_message_column_with_labels)
        train_dataset_chatml = train_dataset_chatml.map(self.format_dataset_chatml)
        # Remove all columns except 'text' to avoid conflicts with SFTTrainer's label handling
        train_dataset_chatml = train_dataset_chatml.remove_columns(
            [col for col in train_dataset_chatml.column_names if col != "text"]
        )
        
        val_dataset_chatml = val_data.map(self.create_message_column_with_labels)
        val_dataset_chatml = val_dataset_chatml.map(self.format_dataset_chatml)
        # Remove all columns except 'text' to avoid conflicts with SFTTrainer's label handling
        val_dataset_chatml = val_dataset_chatml.remove_columns(
            [col for col in val_dataset_chatml.column_names if col != "text"]
        )
        
        return train_dataset_chatml, val_dataset_chatml

    def train_model(self):
        """Train the model using SFTTrainer"""
        from trl import SFTTrainer
        
        # Setup model configuration
        compute_dtype, attn_implementation = self.setup_model_config()
        
        # Prepare datasets
        train_dataset, val_dataset = self.prepare_datasets()
        
        # Setup training arguments (using SFTConfig with assistant_only_loss=True)
        args = self.setup_training_args(compute_dtype)
        
        # Create trainer
        trainer = SFTTrainer(
            model=self.model,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            peft_config=self.peft_config,
            processing_class=self.tokenizer,  # Changed from tokenizer to processing_class
            args=args,
        )
        
        # Train the model
        trainer.train()
        
        # Save the model
        trainer.save_model()
        
        return trainer

    def evaluate_model(self, test_data_path, num_samples=50, create_message_column_with_labels=True):
        """Evaluate the model and compute metrics"""
        import evaluate
        from bert_score import score
        import numpy as np
        from transformers import pipeline
        
        # Load test data
        from datasets import load_dataset
        test_data = load_dataset('csv', data_files=test_data_path, split='train')
        if create_message_column_with_labels:
            test_dataset_chatml = test_data.map(self.create_message_column_with_labels)
        else:
            test_dataset_chatml = test_data.map(self.create_message_column_without_labels)
        test_dataset_chatml = test_dataset_chatml.map(self.format_dataset_chatml)
        
        # Split for testing
        test_split = test_dataset_chatml.train_test_split(test_size=0.25)
        
        # Setup pipeline for inference
        pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)
        
        # Load ROUGE metric
        metric = evaluate.load("rouge")
        
        # Generate predictions
        num_samples = min(num_samples, len(test_split['test']))
        prompts = [
            pipe.tokenizer.apply_chat_template(
                [{"role": "user", "content": test_split['test'][i]['messages'][0]['content']}], 
                tokenize=False, 
                add_generation_prompt=True
            ) for i in range(num_samples)
        ]
        
        outputs = pipe(
            prompts, 
            batch_size=1, 
            max_new_tokens=100, 
            do_sample=True, 
            num_beams=1, 
            temperature=0.7, 
            top_k=50, 
            top_p=0.9,
            repetition_penalty=1.2,
            max_time=180
        )
        
        # Extract predictions
        preds = []
        for i in range(len(outputs)):
            generated_text = outputs[i][0]['generated_text']
            response = generated_text[len(prompts[i]):].split()
            if len(response) > 1:
                pred = " ".join(response)
            else:
                pred = ""
            preds.append(pred)
        
        # Get references
        references = [test_split['test'][i]['counter_argument'] for i in range(len(outputs))]
        tweets = [test_split['test'][i]['text'] for i in range(len(outputs))]
        
        # Compute ROUGE scores
        metric.add_batch(predictions=preds, references=references)
        rouge_result = metric.compute(use_stemmer=True)
        
        # Compute BERTScore
        P, R, F1 = score(preds, references, lang="en", verbose=True)
        
        # Print results
        print("ROUGE Scores:")
        print(f"ROUGE-1: {np.mean(rouge_result['rouge1']) * 100:.2f}")
        print(f"ROUGE-2: {np.mean(rouge_result['rouge2']) * 100:.2f}")
        print(f"ROUGE-L: {np.mean(rouge_result['rougeL']) * 100:.2f}")
        print(f"ROUGE-Lsum: {np.mean(rouge_result['rougeLsum']) * 100:.2f}")
        
        print("\nBERTScore:")
        print(f"Precision: {P.mean().item():.4f}")
        print(f"Recall: {R.mean().item():.4f}")
        print(f"F1: {F1.mean().item():.4f}")
        
        # Save results to CSV
        import pandas as pd
        df = pd.DataFrame(list(zip(tweets, preds, references)), 
                         columns=['tweets', 'predictions', 'references'])
        results_path = f"{self.save_dir}/evaluation_results.csv"
        df.to_csv(results_path, index=False)
        print(f"Results saved to: {results_path}")
        
        return rouge_result, P, R, F1

    def run_training_pipeline(self):
        """Run the complete training pipeline"""
        print("Starting training pipeline...")
        
        # Load data
        self.load_data()
        print("Data loaded successfully")
        
        # Load model
        self.load_model()
        print("Model loaded successfully")
        
        # Train model
        trainer = self.train_model()
        print("Training completed successfully")
        
        return trainer



# Example usage
if __name__ == "__main__":
    # Initialize the model with parameters
    # add argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--dataset_path", type=str, default="/home/sohampoddar/Restored Data/HDD2/sohampoddar/utsav/Data/train_CA_with_label_desc.csv")
    parser.add_argument("--test_data_path", type=str, default="/home/sohampoddar/Restored Data/HDD2/sohampoddar/utsav/Data/val_CA_with_label_desc.csv")
    parser.add_argument("--create_message_column_with_labels", type=bool, default=True)
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--save_dir", type=str, default="/home/sohampoddar/HDD2/utsav/All_Models/Model_Codes_with_labels/llama-3.2-3B-fine-tuned")
    args = parser.parse_args()
    model_name = args.model_name
    dataset_path = args.dataset_path
    test_data_path = args.test_data_path
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    save_dir = args.save_dir
    create_message_column_with_labels = args.create_message_column_with_labels

    model = Finetune_Model(
        model_name=model_name,
        dataset_path=dataset_path,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        save_dir=save_dir
    )
    # Run the complete training pipeline
    trainer = model.run_training_pipeline()
    print("Training completed successfully!")

    # evaluate the model
    rouge_result, P, R, F1 = model.evaluate_model(test_data_path, num_samples=50, create_message_column_with_labels=create_message_column_with_labels)
    print("Evaluation completed successfully!")
    print("ROUGE Scores:")
    print(f"ROUGE-1: {np.mean(rouge_result['rouge1']) * 100:.2f}")
    print(f"ROUGE-2: {np.mean(rouge_result['rouge2']) * 100:.2f}")
    print(f"ROUGE-L: {np.mean(rouge_result['rougeL']) * 100:.2f}")
    print(f"ROUGE-Lsum: {np.mean(rouge_result['rougeLsum']) * 100:.2f}")
    
    print("\nBERTScore:")
    print(f"Precision: {P.mean().item():.4f}")
    print(f"Recall: {R.mean().item():.4f}")
    print(f"F1: {F1.mean().item():.4f}")

    print("Training and evaluation completed successfully!")



# example command to terminal 
# python train.py --model_name "meta-llama/Llama-3.2-3B-Instruct" --dataset_path "/home/sohampoddar/Restored Data/HDD2/sohampoddar/utsav/Data/train_CA_with_label_desc.csv" --test_data_path "/home/sohampoddar/Restored Data/HDD2/sohampoddar/utsav/Data/val_CA_with_label_desc.csv" --create_message_column_with_labels True --num_epochs 2 --learning_rate 1e-4 --weight_decay 0.01 --save_dir "/home/sohampoddar/Restored Data/HDD2/sohampoddar/utsav/All_Models/Fine_Tuned_Models/Llama-3.2-3B-fine-tuned"

# "google/gemma-2b-it"
# python train.py --model_name "google/gemma-2b-it" --dataset_path "/home/sohampoddar/Restored Data/HDD2/sohampoddar/utsav/Data/train_CA_with_label_desc.csv" --test_data_path "/home/sohampoddar/Restored Data/HDD2/sohampoddar/utsav/Data/val_CA_with_label_desc.csv" --create_message_column_with_labels True --num_epochs 2 --learning_rate 1e-4 --weight_decay 0.01 --save_dir "/home/sohampoddar/Restored Data/HDD2/sohampoddar/utsav/All_Models/Fine_Tuned_Models/gemma-2b-it-fine-tuned"


# "microsoft/Phi-3-mini-4k-instruct"
# python train.py --model_name "microsoft/Phi-3-mini-4k-instruct" --dataset_path "/home/sohampoddar/Restored Data/HDD2/sohampoddar/utsav/Data/train_CA_with_label_desc.csv" --test_data_path "/home/sohampoddar/Restored Data/HDD2/sohampoddar/utsav/Data/val_CA_with_label_desc.csv" --create_message_column_with_labels True --num_epochs 2 --learning_rate 1e-4 --weight_decay 0.01 --save_dir "/home/sohampoddar/Restored Data/HDD2/sohampoddar/utsav/All_Models/Fine_Tuned_Models/Phi-3-mini-4k-instruct-fine-tuned"
