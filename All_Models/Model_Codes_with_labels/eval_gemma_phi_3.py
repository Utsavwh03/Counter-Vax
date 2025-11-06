#!/usr/bin/env python3
"""
Test script for fine-tuned models on a single sample or full evaluation.
Supports both Gemma and Llama models.
"""

import os
import torch
import argparse
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from datasets import load_dataset
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


class SingleSampleTester:
    """Class to test fine-tuned models on a single sample"""
    
    def __init__(self, model_path, base_model_name=None, use_quantization=True):
        """
        Initialize the tester.
        
        Args:
            model_path (str): Path to the fine-tuned model directory
            base_model_name (str, optional): Base model name if loading a PEFT model
            use_quantization (bool): Whether to use 4-bit quantization
        """
        self.model_path = model_path
        self.base_model_name = base_model_name
        self.use_quantization = use_quantization
        self.model = None
        self.tokenizer = None
        
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
    
    def create_prompt(self, tweet, labels=None, use_labels=True):
        """
        Create prompt from tweet and optional labels.
        
        Args:
            tweet (str): The input tweet text
            labels (str or list, optional): Labels for the tweet
            use_labels (bool): Whether to include labels in the prompt
            
        Returns:
            dict: Messages dict formatted for chat template
        """
        if use_labels and labels:
            prompt = f"Generate Counter Argument for the anti-vaccine tweet:\n Tweet: {tweet}\n Talk About "
            
            if isinstance(labels, str):
                lab = labels.split()
            else:
                lab = labels
            
            if isinstance(lab, list):
                mapped_labels = " and ".join([labels_map.get(l, "") for l in lab if l])
            else:
                mapped_labels = labels_map.get(lab, "")
            
            prompt += mapped_labels
            prompt += " ##Output: "
        else:
            prompt = f"Generate Counter Argument for the anti-vaccine tweet:\n Tweet: {tweet}\n ##Output: "
        
        messages = [{"role": "user", "content": prompt}]
        return messages
    
    def generate(self, tweet, labels=None, use_labels=True, max_new_tokens=150, 
                 temperature=0.7, top_k=50, top_p=0.9, repetition_penalty=1.2):
        """
        Generate counter argument for a single tweet.
        
        Args:
            tweet (str): The input tweet text
            labels (str or list, optional): Labels for the tweet
            use_labels (bool): Whether to include labels in the prompt
            max_new_tokens (int): Maximum tokens to generate
            temperature (float): Sampling temperature
            top_k (int): Top-k sampling
            top_p (float): Top-p sampling
            repetition_penalty (float): Repetition penalty
            
        Returns:
            str: Generated counter argument
        """
        if self.model is None or self.tokenizer is None:
            self.load_model()
        
        # Create prompt
        messages = self.create_prompt(tweet, labels, use_labels)
        
        # Apply chat template
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        
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
        
        # Decode only the generated part
        generated_tokens = outputs[0, inputs['input_ids'].shape[1]:]
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return generated_text.strip()
    
    def test_from_dict(self, sample_dict, use_labels=True, **generation_kwargs):
        """
        Test on a sample dictionary (like a row from a CSV).
        
        Args:
            sample_dict (dict): Dictionary with 'text' and optionally 'labels' keys
            use_labels (bool): Whether to use labels if available
            **generation_kwargs: Additional arguments for generation
            
        Returns:
            str: Generated counter argument
        """
        tweet = sample_dict.get('text', '')
        labels = sample_dict.get('labels', None) if use_labels else None
        
        return self.generate(tweet, labels, use_labels, **generation_kwargs)
    
    def evaluate_dataset(self, test_data_path, use_labels=True, num_samples=None,
                        batch_size=8, max_new_tokens=150, temperature=0.7, 
                        top_k=50, top_p=0.9, repetition_penalty=1.2, 
                        save_results_path=None):
        """
        Evaluate the model on a full dataset.
        
        Args:
            test_data_path (str): Path to test data CSV file
            use_labels (bool): Whether to use labels in prompts
            num_samples (int, optional): Number of samples to evaluate (None for all)
            batch_size (int): Batch size for inference
            max_new_tokens (int): Maximum tokens to generate
            temperature (float): Sampling temperature
            top_k (int): Top-k sampling
            top_p (float): Top-p sampling
            repetition_penalty (float): Repetition penalty
            save_results_path (str, optional): Path to save results CSV
            
        Returns:
            dict: Evaluation results with metrics
        """
        if self.model is None or self.tokenizer is None:
            self.load_model()
        
        print(f"\n{'='*70}")
        print(f"Evaluating Model on Dataset")
        print(f"{'='*70}")
        print(f"Test data: {test_data_path}")
        print(f"Using labels: {use_labels}")
        print(f"Batch size: {batch_size}")
        print(f"{'='*70}\n")
        
        # Load dataset
        test_data = load_dataset('csv', data_files=test_data_path, split='train')
        
        # Determine number of samples
        if num_samples is None:
            num_samples = len(test_data)
        else:
            num_samples = min(num_samples, len(test_data))
        
        print(f"Evaluating on {num_samples} samples...\n")
        
        # Prepare all prompts and references
        prompts = []
        references = []
        tweets = []
        all_labels = []
        
        for i in tqdm(range(num_samples), desc="Preparing prompts"):
            row = test_data[i]
            tweet = row.get('text', '')
            labels = row.get('labels', None) if use_labels else None
            counter_argument = row.get('counter_argument', '')
            
            messages = self.create_prompt(tweet, labels, use_labels)
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            prompts.append(formatted_prompt)
            references.append(counter_argument)
            tweets.append(tweet)
            all_labels.append(labels)
        
        # Generate predictions in batches
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
            self.tokenizer.padding_side = original_padding_side
            
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
            
            # Decode batch
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
                # Or more accurately, from actual_input_len position in the sequence
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
        
        # Compute metrics
        print("\nComputing metrics...")
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
        
        bertscore_precision = P.mean().item()
        bertscore_recall = R.mean().item()
        bertscore_f1 = F1.mean().item()
        
        # Print results
        print(f"\n{'='*70}")
        print("EVALUATION RESULTS")
        print(f"{'='*70}")
        print("ROUGE Scores:")
        print(f"  ROUGE-1: {rouge1_mean:.2f}")
        print(f"  ROUGE-2: {rouge2_mean:.2f}")
        print(f"  ROUGE-L: {rougeL_mean:.2f}")
        print(f"  ROUGE-Lsum: {rougeLsum_mean:.2f}")
        print("\nBERTScore:")
        print(f"  Precision: {bertscore_precision:.4f}")
        print(f"  Recall: {bertscore_recall:.4f}")
        print(f"  F1: {bertscore_f1:.4f}")
        print(f"{'='*70}\n")
        
        # Save results
        results = {
            'tweet': tweets,
            'prediction': predictions,
            'reference': references,
            'labels': all_labels if use_labels else ['N/A'] * len(tweets)
        }
        
        df = pd.DataFrame(results)
        
        if save_results_path:
            df.to_csv(save_results_path, index=False)
            print(f"Results saved to: {save_results_path}\n")
        else:
            # Auto-generate filename
            label_suffix = "with_labels" if use_labels else "without_labels"
            base_name = os.path.splitext(os.path.basename(test_data_path))[0]
            model_name = os.path.basename(self.model_path.rstrip('/'))
            save_path = f"All_Models/Eval results/{label_suffix}/{model_name}_{label_suffix}.csv"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            df.to_csv(save_path, index=False)
            print(f"Results saved to: {save_path}\n")
        
        return {
            'rouge1': rouge1_mean,
            'rouge2': rouge2_mean,
            'rougeL': rougeL_mean,
            'rougeLsum': rougeLsum_mean,
            'bertscore_precision': bertscore_precision,
            'bertscore_recall': bertscore_recall,
            'bertscore_f1': bertscore_f1,
            'predictions': predictions,
            'references': references
        }


def test_single_sample(model_path, tweet, labels=None, use_labels=True, 
                       base_model_name=None, use_quantization=True, **generation_kwargs):
    """
    Convenience function to test a single sample.
    
    Args:
        model_path (str): Path to fine-tuned model
        tweet (str): Tweet text to generate counter argument for
        labels (str or list, optional): Labels for the tweet
        use_labels (bool): Whether to include labels in prompt
        base_model_name (str, optional): Base model name for PEFT models
        use_quantization (bool): Whether to use quantization
        **generation_kwargs: Generation parameters (max_new_tokens, temperature, etc.)
        
    Returns:
        str: Generated counter argument
    """
    tester = SingleSampleTester(model_path, base_model_name, use_quantization)
    return tester.generate(tweet, labels, use_labels, **generation_kwargs)


def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(description="Test fine-tuned model on a single sample or evaluate on full dataset")
    
    # Mode selection
    parser.add_argument("--mode", type=str, choices=["single", "evaluate"], default="single",
                       help="Mode: 'single' for single sample test, 'evaluate' for full dataset evaluation")
    
    # Common arguments
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to fine-tuned model directory")
    parser.add_argument("--base_model_name", type=str, default=None,
                       help="Base model name if loading PEFT model")
    parser.add_argument("--use_quantization", type=bool, default=True,
                       help="Whether to use 4-bit quantization")
    
    # Single sample mode arguments
    parser.add_argument("--tweet", type=str, default=None,
                       help="Tweet text to test (required for single mode)")
    parser.add_argument("--labels", type=str, default=None,
                       help="Labels for the tweet (space-separated)")
    parser.add_argument("--use_labels", type=bool, default=True,
                       help="Whether to use labels in prompt")
    
    # Evaluation mode arguments
    parser.add_argument("--test_data_path", type=str, default=None,
                       help="Path to test data CSV file (required for evaluate mode)")
    parser.add_argument("--num_samples", type=int, default=None,
                       help="Number of samples to evaluate (None for all)")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size for evaluation")
    parser.add_argument("--save_results_path", type=str, default=None,
                       help="Path to save results CSV (auto-generated if not provided)")
    
    # Generation parameters
    parser.add_argument("--max_new_tokens", type=int, default=150,
                       help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50,
                       help="Top-k sampling")
    parser.add_argument("--top_p", type=float, default=0.9,
                       help="Top-p sampling")
    parser.add_argument("--repetition_penalty", type=float, default=1.2,
                       help="Repetition penalty")
    
    args = parser.parse_args()
    
    if args.mode == "single":
        # Single sample mode
        if args.tweet is None:
            parser.error("--tweet is required for single mode")
        
        labels = args.labels.split() if args.labels else None
        
        print("=" * 70)
        print("Testing Fine-Tuned Model on Single Sample")
        print("=" * 70)
        print(f"\nModel Path: {args.model_path}")
        print(f"\nTweet: {args.tweet}")
        if labels:
            print(f"Labels: {labels}")
        print("\nGenerating counter argument...")
        print("-" * 70)
        
        result = test_single_sample(
            model_path=args.model_path,
            tweet=args.tweet,
            labels=labels,
            use_labels=args.use_labels,
            base_model_name=args.base_model_name,
            use_quantization=args.use_quantization,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty
        )
        
        print(f"\nGenerated Counter Argument:\n{result}")
        print("=" * 70)
        
        return result
    
    else:
        # Evaluation mode
        if args.test_data_path is None:
            parser.error("--test_data_path is required for evaluate mode")
        
        tester = SingleSampleTester(args.model_path, args.base_model_name, args.use_quantization)
        
        # Evaluate with labels
        print("\n" + "="*70)
        print("EVALUATION WITH LABELS")
        print("="*70)
        results_with_labels = tester.evaluate_dataset(
            test_data_path=args.test_data_path,
            use_labels=True,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            save_results_path=(args.save_results_path.replace('.csv', '_with_labels.csv') if args.save_results_path and args.save_results_path.endswith('.csv') 
                               else (args.save_results_path + '_with_labels.csv' if args.save_results_path else None))
        )
        
        # Evaluate without labels
        print("\n" + "="*70)
        print("EVALUATION WITHOUT LABELS")
        print("="*70)
        results_without_labels = tester.evaluate_dataset(
            test_data_path=args.test_data_path,
            use_labels=False,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            save_results_path=(args.save_results_path.replace('.csv', '_without_labels.csv') if args.save_results_path and args.save_results_path.endswith('.csv')
                               else (args.save_results_path + '_without_labels.csv' if args.save_results_path else None))
        )
        
        # Print comparison
        print("\n" + "="*70)
        print("COMPARISON SUMMARY")
        print("="*70)
        print("\nWith Labels:")
        print(f"  ROUGE-1: {results_with_labels['rouge1']:.2f}")
        print(f"  ROUGE-L: {results_with_labels['rougeL']:.2f}")
        print(f"  BERTScore F1: {results_with_labels['bertscore_f1']:.4f}")
        print("\nWithout Labels:")
        print(f"  ROUGE-1: {results_without_labels['rouge1']:.2f}")
        print(f"  ROUGE-L: {results_without_labels['rougeL']:.2f}")
        print(f"  BERTScore F1: {results_without_labels['bertscore_f1']:.4f}")
        print("="*70 + "\n")
        
        return {
            'with_labels': results_with_labels,
            'without_labels': results_without_labels
        }


if __name__ == "__main__":
    main()


# Example usage:
# python eval_gemma_phi_3.py --model_path "/path/to/model" --tweet "Your tweet here" --use_labels True

# python eval_gemma_phi_3.py --model_path "/home/sohampoddar/Restored Data/HDD2/sohampoddar/utsav/All_Models/Fine_Tuned_Models/Phi-3-mini-4k-instruct-fine-tuned" --tweet "Another Covid 19 vaccine casualty Illaria Pappa 31 school teacher from #Italy died from thromboembolism 10 days after receiving #AstraZeneca #vaccine" --use_labels True

# test gemma on full dataset on val_CA_with_label_desc_with_predicted_labels_from_covid_bert.csv
# python eval_gemma_phi_3.py --mode evaluate --model_path "/home/sohampoddar/Restored Data/HDD2/sohampoddar/utsav/All_Models/Fine_Tuned_Models/gemma-2b-it-fine-tuned" --test_data_path "/home/sohampoddar/Restored Data/HDD2/sohampoddar/utsav/Data/val_CA_with_label_desc_with_predicted_labels_from_covid_bert.csv" --use_labels True --save_results_path "/home/sohampoddar/Restored Data/HDD2/sohampoddar/utsav/All_Models/Eval results/With Labels From Two Step/gemma2b_with_predicted_labels_from_covid_bert.csv"

# test phi-3-mini-4k-instruct on full dataset on val_CA_with_label_desc_with_predicted_labels_from_covid_bert.csv
# python eval_gemma_phi_3.py --mode evaluate --model_path "/home/sohampoddar/Restored Data/HDD2/sohampoddar/utsav/All_Models/Fine_Tuned_Models/Phi-3-mini-4k-instruct-fine-tuned" --test_data_path "/home/sohampoddar/Restored Data/HDD2/sohampoddar/utsav/Data/val_CA_with_label_desc_with_predicted_labels_from_covid_bert.csv" --use_labels True --save_results_path "/home/sohampoddar/Restored Data/HDD2/sohampoddar/utsav/All_Models/Eval results/With Labels From Two Step/phi3mini4k_with_predicted_labels_from_covid_bert.csv"

# test phi-3-mini-4k-instruct on full dataset on val_CA_with_label_desc_with_predicted_labels_from_t5.csv
# python eval_gemma_phi_3.py --mode evaluate --model_path "/home/sohampoddar/Restored Data/HDD2/sohampoddar/utsav/All_Models/Fine_Tuned_Models/Phi-3-mini-4k-instruct-fine-tuned" --test_data_path "/home/sohampoddar/Restored Data/HDD2/sohampoddar/utsav/Data/val_CA_with_label_desc_with_predicted_labels_from_t5.csv" --use_labels True --save_results_path "/home/sohampoddar/Restored Data/HDD2/sohampoddar/utsav/All_Models/Eval results/With Labels From T5base/phi3mini4k_with_predicted_labels_from_t5.csv"

# test gemma on full dataset on val_CA_with_label_desc_with_predicted_labels_from_t5.csv
# python eval_gemma_phi_3.py --mode evaluate --model_path "/home/sohampoddar/Restored Data/HDD2/sohampoddar/utsav/All_Models/Fine_Tuned_Models/gemma-2b-it-fine-tuned" --test_data_path "/home/sohampoddar/Restored Data/HDD2/sohampoddar/utsav/Data/val_CA_with_label_desc_with_predicted_labels_from_t5.csv" --use_labels True --save_results_path "/home/sohampoddar/Restored Data/HDD2/sohampoddar/utsav/All_Models/Eval results/With Labels From T5base/gemma2b_with_predicted_labels_from_t5.csv"