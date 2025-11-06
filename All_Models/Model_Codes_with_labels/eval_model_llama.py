import os
import gc
import torch
import pandas as pd
import numpy as np
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import json
import evaluate
from bert_score import score
from tqdm import tqdm

# LABEL DESCRIPTIONS (same as in train.py)
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


class EvalModel:
    """
    A class for evaluating fine-tuned language models.
    Supports loading models from Hugging Face Hub or local paths.
    Uses the same logic as test_single_sample.py for reliable evaluation.
    """
    
    def __init__(self, model_path, base_model_name=None, use_quantization=True):
        """
        Initialize the EvalModel class.
        
        Args:
            model_path (str): Path to the fine-tuned model (local path or Hugging Face model ID)
            base_model_name (str, optional): Base model name if loading a PEFT model
            use_quantization (bool): Whether to use 4-bit quantization for memory efficiency
        """
        self.model_path = model_path
        self.base_model_name = base_model_name
        self.use_quantization = use_quantization
        self.model = None
        self.tokenizer = None
        
        # Setup device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f"Using GPU: {torch.cuda.get_device_name(self.device)}")
        else:
            self.device = torch.device("cpu")
            print("Using CPU")
    
    def load_model(self):
        """Load the fine-tuned model and tokenizer"""
        print(f"Loading model from: {self.model_path}")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            print("✅ Tokenizer loaded successfully")
            
            # Setup tokenizer
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)
            
            # Auto-detect PEFT model if base_model_name not provided
            adapter_config_path = os.path.join(self.model_path, "adapter_config.json")
            if not self.base_model_name and os.path.exists(adapter_config_path):
                with open(adapter_config_path, 'r') as f:
                    adapter_config = json.load(f)
                    self.base_model_name = adapter_config.get("base_model_name_or_path")
                    if self.base_model_name:
                        print(f"✅ Auto-detected PEFT model. Base model: {self.base_model_name}")
            
            # Determine compute dtype
            if torch.cuda.is_bf16_supported():
                compute_dtype = torch.bfloat16
            else:
                compute_dtype = torch.float16
            
            # Load model
            if self.use_quantization:
                # Setup quantization config
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16
                )
                
                # Load base model if PEFT model
                if self.base_model_name:
                    base_model = AutoModelForCausalLM.from_pretrained(
                        self.base_model_name,
                        torch_dtype=compute_dtype,
                        trust_remote_code=True,
                        quantization_config=bnb_config,
                        device_map="auto"
                    )
                    # Load PEFT model
                    self.model = PeftModel.from_pretrained(base_model, self.model_path)
                else:
                    # Load regular fine-tuned model
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_path,
                        torch_dtype=compute_dtype,
                        trust_remote_code=True,
                        quantization_config=bnb_config,
                        device_map="auto"
                    )
            else:
                # Load without quantization
                if self.base_model_name:
                    base_model = AutoModelForCausalLM.from_pretrained(
                        self.base_model_name,
                        torch_dtype=compute_dtype,
                        trust_remote_code=True,
                        device_map="auto"
                    )
                    self.model = PeftModel.from_pretrained(base_model, self.model_path)
                else:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_path,
                        torch_dtype=compute_dtype,
                        trust_remote_code=True,
                        device_map="auto"
                    )
            
            print("✅ Model loaded successfully")
            print(f"✅ Model device: {next(self.model.parameters()).device}")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def create_message_column_with_labels(self, row):
        """Create message format with labels for evaluation"""
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
        messages.append(user)
        
        assistant = {
            "content": f"{row['counter_argument']}",
            "role": "assistant"
        }
        messages.append(assistant)
        
        return {"messages": messages}
    
    def create_message_column_without_labels(self, row):
        """Create message format without labels for evaluation"""
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
    
    def load_test_data(self, test_data_path, create_message_column_with_labels=True):
        """Load and prepare test data"""
        print(f"Loading test data from: {test_data_path}")
        
        # Load dataset
        test_data = load_dataset('csv', data_files=test_data_path, split='train')
        
        # Process dataset
        if create_message_column_with_labels:
            test_dataset_chatml = test_data.map(self.create_message_column_with_labels)
        else:
            test_dataset_chatml = test_data.map(self.create_message_column_without_labels)
        
        print(f"✅ Test data loaded: {len(test_dataset_chatml)} samples")
        return test_dataset_chatml
    
    def evaluate_model(self, test_data_path, num_samples=50, create_message_column_with_labels=True, 
                      max_new_tokens=100, temperature=0.7, top_k=50, top_p=0.9, repetition_penalty=1.2,
                      batch_size=8, save_results_path=None):
        """
        Evaluate the model on test data and compute metrics using batched inference.
        Uses the same logic as test_single_sample.py for reliable token extraction.
        
        Args:
            test_data_path (str): Path to test data CSV file
            num_samples (int, optional): Number of samples to evaluate (None for all)
            create_message_column_with_labels (bool): Whether to use labels in prompts
            max_new_tokens (int): Maximum tokens to generate
            temperature (float): Sampling temperature
            top_k (int): Top-k sampling
            top_p (float): Top-p sampling
            repetition_penalty (float): Repetition penalty
            batch_size (int): Batch size for batched inference (default: 8)
            save_results_path (str, optional): Path to save results CSV
            
        Returns:
            dict: Evaluation results including ROUGE and BERTScore metrics
        """
        print("Starting model evaluation with batched inference...")
        
        # Load test data
        test_dataset = self.load_test_data(test_data_path, create_message_column_with_labels)
        
        # Determine number of samples
        if num_samples is None:
            num_samples = len(test_dataset)
        else:
            num_samples = min(num_samples, len(test_dataset))
        
        print(f"Evaluating on {num_samples} samples with batch_size={batch_size}...")
        
        # Prepare all prompts and references
        print("Preparing prompts...")
        prompts = []
        references = []
        tweets = []
        
        for i in tqdm(range(num_samples), desc="Preparing prompts"):
            # Format prompt using chat template
            messages = test_dataset[i]['messages']
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            prompts.append(formatted_prompt)
            references.append(test_dataset[i]['counter_argument'])
            tweets.append(test_dataset[i]['text'])
        
        # Generate predictions in batches (using exact logic from test_single_sample.py)
        print(f"\nGenerating predictions in batches of {batch_size}...")
        predictions = []
        
        for batch_idx in tqdm(range(0, len(prompts), batch_size), desc="Generating"):
            batch_prompts = prompts[batch_idx:batch_idx + batch_size]
            
            # Set padding side to 'left' for decoder-only models (required for batched generation)
            original_padding_side = self.tokenizer.padding_side
            self.tokenizer.padding_side = 'left'
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            
            # Restore original padding side
            # self.tokenizer.padding_side = original_padding_side
            
            # Move to device
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=False,  # Disable KV cache to avoid DynamicCache issues
                )
            
            # Decode batch (EXACT LOGIC FROM test_single_sample.py)
            # With left padding, inputs are at the end of the sequence
            # Each input in the batch may have different lengths due to padding
            batch_size_actual = outputs.shape[0]
            max_input_length = inputs['input_ids'].shape[1]
            input_lengths = inputs['attention_mask'].sum(dim=1).cpu().tolist()
            
            for i in range(batch_size_actual):
                # Get actual input length (excluding padding)
                actual_input_len = input_lengths[i]
                
                # With left padding: actual input tokens are at the end of the padded sequence
                # Generated tokens start right after the input tokens
                # outputs[i] contains: [pad_tokens...][input_tokens][generated_tokens]
                # We need to extract from position max_input_length onwards (which contains generated tokens)
                generated_tokens = outputs[i, max_input_length:].cpu()
                
                # Alternative: decode full output and remove input
                full_output = outputs[i].cpu()
                full_decoded = self.tokenizer.decode(full_output, skip_special_tokens=True)
                
                # Decode just the input to get the prompt text (accounting for left padding)
                # For left padding, we need to decode only the actual input tokens
                input_ids_actual = inputs['input_ids'][i, max_input_length - actual_input_len:max_input_length]
                input_decoded = self.tokenizer.decode(input_ids_actual.cpu(), skip_special_tokens=True)
                
                # Extract generated part by removing input from output
                if full_decoded.endswith(input_decoded):
                    # Input is at the end, so remove it
                    decoded = full_decoded[:-len(input_decoded)].strip()
                elif full_decoded.startswith(input_decoded):
                    # Fallback: input at start
                    decoded = full_decoded[len(input_decoded):].strip()
                else:
                    # Use token slice method as fallback
                    decoded = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
                
                predictions.append(decoded if decoded else "")
        
        # Compute ROUGE scores
        print("\nComputing ROUGE scores...")
        metric = evaluate.load("rouge")
        rouge_result = metric.compute(predictions=predictions, references=references, use_stemmer=True)
        
        # Compute BERTScore
        print("Computing BERTScore...")
        P, R, F1 = score(predictions, references, lang="en", verbose=True)
        
        # Calculate mean scores
        rouge1_mean = np.mean(rouge_result['rouge1']) * 100
        rouge2_mean = np.mean(rouge_result['rouge2']) * 100
        rougeL_mean = np.mean(rouge_result['rougeL']) * 100
        rougeLsum_mean = np.mean(rouge_result['rougeLsum']) * 100
        
        bert_precision = P.mean().item()
        bert_recall = R.mean().item()
        bert_f1 = F1.mean().item()
        
        # Print results
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        print("ROUGE Scores:")
        print(f"ROUGE-1: {rouge1_mean:.2f}")
        print(f"ROUGE-2: {rouge2_mean:.2f}")
        print(f"ROUGE-L: {rougeL_mean:.2f}")
        print(f"ROUGE-Lsum: {rougeLsum_mean:.2f}")
        
        print("\nBERTScore:")
        print(f"Precision: {bert_precision:.4f}")
        print(f"Recall: {bert_recall:.4f}")
        print(f"F1: {bert_f1:.4f}")
        print("="*50)
        
        # Save results to CSV
        results_df = pd.DataFrame({
            'tweets': tweets,
            'predictions': predictions,
            'references': references
        })
        
        # Use provided save_results_path or default to model_path
        if save_results_path:
            results_path = save_results_path
        else:
            results_path = f"{self.model_path}/evaluation_results.csv"
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        results_df.to_csv(results_path, index=False)
        print(f"Results saved to: {results_path}")
        
        # Return comprehensive results
        results = {
            'rouge': {
                'rouge1': rouge1_mean,
                'rouge2': rouge2_mean,
                'rougeL': rougeL_mean,
                'rougeLsum': rougeLsum_mean
            },
            'bert': {
                'precision': bert_precision,
                'recall': bert_recall,
                'f1': bert_f1
            },
            'predictions': predictions,
            'references': references,
            'tweets': tweets
        }
        
        return results
    
    def run_evaluation(self, test_data_path, num_samples=None, create_message_column_with_labels=True, 
                      save_results_path=None, **kwargs):
        """
        Run complete evaluation pipeline
        
        Args:
            test_data_path (str): Path to test data CSV file
            num_samples (int, optional): Number of samples to evaluate
            create_message_column_with_labels (bool): Whether to use labels in prompts
            save_results_path (str, optional): Custom path to save results CSV
            **kwargs: Additional arguments for evaluation
            
        Returns:
            dict: Evaluation results
        """
        print("Starting evaluation pipeline...")
        
        # Load model if not already loaded
        if self.model is None:
            self.load_model()
        
        # Run evaluation
        results = self.evaluate_model(
            test_data_path=test_data_path,
            num_samples=num_samples,
            create_message_column_with_labels=create_message_column_with_labels,
            save_results_path=save_results_path,
            **kwargs
        )
        
        print("Evaluation completed successfully!")
        return results


def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned language model")
    parser.add_argument("--model_path", type=str, required=True, 
                       help="Path to the fine-tuned model")
    parser.add_argument("--base_model_name", type=str, default=None,
                       help="Base model name if loading a PEFT model")
    parser.add_argument("--test_data_path", type=str, required=True,
                       help="Path to test data CSV file")
    parser.add_argument("--num_samples", type=int, default=None,
                       help="Number of samples to evaluate (None for all)")
    parser.add_argument("--create_message_column_with_labels", type=bool, default=True,
                       help="Whether to use labels in prompts")
    parser.add_argument("--use_quantization", type=bool, default=True,
                       help="Whether to use 4-bit quantization")
    parser.add_argument("--max_new_tokens", type=int, default=100,
                       help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50,
                       help="Top-k sampling")
    parser.add_argument("--top_p", type=float, default=0.9,
                       help="Top-p sampling")
    parser.add_argument("--repetition_penalty", type=float, default=1.2,
                       help="Repetition penalty")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size for batched inference (default: 8)")
    parser.add_argument("--save_results_path", type=str, default=None,
                       help="Path to save the results to a CSV file")
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = EvalModel(
        model_path=args.model_path,
        base_model_name=args.base_model_name,
        use_quantization=args.use_quantization
    )
    
    # Run evaluation
    results = evaluator.run_evaluation(
        test_data_path=args.test_data_path,
        num_samples=args.num_samples,
        create_message_column_with_labels=args.create_message_column_with_labels,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        batch_size=args.batch_size,
        save_results_path=args.save_results_path
    )
    
    return results


if __name__ == "__main__":
    main()


# Example usage:
# python eval_model_llama.py --model_path "/path/to/fine-tuned/model" --test_data_path "/path/to/test_data.csv" --num_samples 100

# Use labels in the prompt for evaluation
# python eval_model_llama.py --model_path "/home/sohampoddar/Restored Data/HDD2/sohampoddar/utsav/All_Models/Fine_Tuned_Models/Llama-3.2-3B-fine-tuned" --test_data_path "/home/sohampoddar/Restored Data/HDD2/sohampoddar/utsav/Data/val_CA_with_label_desc.csv" --num_samples 100 --batch_size 16 --create_message_column_with_labels True

# Don't use labels in the prompt for evaluation
# python eval_model_llama.py --model_path "/home/sohampoddar/Restored Data/HDD2/sohampoddar/utsav/All_Models/Fine_Tuned_Models/Llama-3.2-3B-fine-tuned" --test_data_path "/home/sohampoddar/Restored Data/HDD2/sohampoddar/utsav/Data/val_CA_with_label_desc.csv" --num_samples 100 --batch_size 16 --create_message_column_with_labels False --save_results_path "/home/sohampoddar/Restored Data/HDD2/sohampoddar/utsav/All_Models/Eval results/Without Labels/llama3.2_without_labels.csv"


# check on val_CA_with_label_desc_with_predicted_labels_from_covid_bert.csv
# python eval_model_llama.py --model_path "/home/sohampoddar/Restored Data/HDD2/sohampoddar/utsav/All_Models/Fine_Tuned_Models/Llama-3.2-3B-fine-tuned" --test_data_path "/home/sohampoddar/Restored Data/HDD2/sohampoddar/utsav/Data/val_CA_with_label_desc_with_predicted_labels_from_covid_bert.csv" --num_samples 100 --batch_size 16 --create_message_column_with_labels True --save_results_path "/home/sohampoddar/Restored Data/HDD2/sohampoddar/utsav/All_Models/Eval results/With Labels/llama3.2_with_predicted_labels_from_covid_bert.csv"

# check on val_CA_with_label_desc_with_predicted_labels_from_t5.csv
# python eval_model_llama.py --model_path "/home/sohampoddar/Restored Data/HDD2/sohampoddar/utsav/All_Models/Fine_Tuned_Models/Llama-3.2-3B-fine-tuned" --test_data_path "/home/sohampoddar/Restored Data/HDD2/sohampoddar/utsav/Data/val_CA_with_label_desc_with_predicted_labels_from_t5.csv" --num_samples 100 --batch_size 16 --create_message_column_with_labels True --save_results_path "/home/sohampoddar/Restored Data/HDD2/sohampoddar/utsav/All_Models/Eval results/With Labels From T5base/llama3.2_with_predicted_labels_from_t5.csv"