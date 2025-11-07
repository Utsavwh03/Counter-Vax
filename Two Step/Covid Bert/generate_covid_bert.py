import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
from pathlib import Path

class TweetDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }


def predict_labels(model_path, tokenizer_path, val_data_path, output_path=None, batch_size=16):
    """
    Predict labels for validation data using the fine-tuned COVID Twitter BERT model.
    
    Args:
        model_path (str): Path to the saved fine-tuned model
        tokenizer_path (str): Path to the saved tokenizer
        val_data_path (str): Path to validation CSV file
        output_path (str, optional): Path to save predictions CSV
        batch_size (int): Batch size for inference
    """
    print("="*70)
    print("COVID Twitter BERT - Label Prediction")
    print("="*70)
    
    # Define label list (must match training)
    label_list = ['rushed', 'side-effect', 'ineffective', 'mandatory', 'pharma', 
                  'ingredients', 'country', 'conspiracy', 'political', 'unnecessary', 'none']
    num_labels = len(label_list)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    # Validate paths exist and convert to absolute paths
    tokenizer_path = os.path.abspath(os.path.expanduser(tokenizer_path))
    model_path = os.path.abspath(os.path.expanduser(model_path))
    
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer path not found: {tokenizer_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path not found: {model_path}")
    
    # Check if paths point to directories (local model files)
    is_local_tokenizer = os.path.isdir(tokenizer_path)
    is_local_model = os.path.isdir(model_path)
    
    # Load tokenizer - if it's a local directory, use local_files_only=True to avoid repo_id validation
    print(f"\nLoading tokenizer from: {tokenizer_path}")
    if is_local_tokenizer:
        # For local directories, use local_files_only=True to bypass HuggingFace Hub validation
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            local_files_only=True
        )
    else:
        # For Hub models, don't use local_files_only
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    print("✅ Tokenizer loaded successfully")
    
    # Load model - if it's a local directory, use local_files_only=True to avoid repo_id validation
    print(f"\nLoading model from: {model_path}")
    if is_local_model:
        # For local directories, use local_files_only=True to bypass HuggingFace Hub validation
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=num_labels,
            problem_type="multi_label_classification",
            local_files_only=True
        )
    else:
        # For Hub models, don't use local_files_only
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=num_labels,
            problem_type="multi_label_classification"
        )
    model.to(device)
    model.eval()
    print("✅ Model loaded successfully")
    
    # Load validation data
    print(f"\nLoading validation data from: {val_data_path}")
    val_data = pd.read_csv(val_data_path)
    print(f"✅ Loaded {len(val_data)} samples")
    
    # Check if labels exist and prepare them
    has_true_labels = 'labels' in val_data.columns
    if has_true_labels:
        # Convert labels to list format for processing
        val_data['labels_list'] = val_data['labels'].apply(
            lambda x: x.split(' ') if isinstance(x, str) else []
        )
        # Keep original labels as string for output
        true_labels_strings = val_data['labels'].tolist()
    else:
        true_labels_strings = [None] * len(val_data)
        val_data['labels_list'] = [[]] * len(val_data)
    
    # Extract texts
    val_texts = val_data['text'].tolist()
    
    # Create dataset and dataloader
    max_len = 512
    val_dataset = TweetDataset(val_texts, tokenizer, max_len)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Predict
    print(f"\nGenerating predictions (batch_size={batch_size})...")
    all_predictions = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Predicting"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask)
            probs = torch.sigmoid(outputs.logits)
            preds = (probs > 0.5).int()
            
            all_predictions.extend(preds.cpu().numpy())
    
    # Convert predictions to label strings
    mlb = MultiLabelBinarizer(classes=label_list)
    # Fit on empty to establish classes
    mlb.fit([[]])
    
    # Convert binary predictions to label strings
    predicted_labels = []
    for pred in all_predictions:
        # Get indices where prediction is 1
        label_indices = [i for i, val in enumerate(pred) if val == 1]
        # Get corresponding label names
        labels = [label_list[i] for i in label_indices]
        # Join as space-separated string (matching original format)
        if labels:
            predicted_labels.append(' '.join(labels))
        else:
            predicted_labels.append('none')
    
    # Create results dataframe
    results_df = pd.DataFrame({
        'text': val_data['text'].values,
        'predicted_labels': predicted_labels
    })
    
    # Add individual predicted label columns (binary)
    for idx, label in enumerate(label_list):
        results_df[f'pred_{label}'] = [pred[idx] for pred in all_predictions]
    
    # If original labels exist, include them for comparison
    if has_true_labels:
        # Add true labels as space-separated string
        results_df['true_labels'] = true_labels_strings
        
        print("\n⚠️  Note: Validation data contains true labels. Comparing predictions...")
        
        # Calculate metrics if true labels exist
        if len(all_predictions) > 0:
            # Convert true labels to binary format
            true_labels_binary = mlb.transform(val_data['labels_list'])
            
            # Add individual true label columns (binary) for comparison
            for idx, label in enumerate(label_list):
                results_df[f'true_{label}'] = [true[idx] for true in true_labels_binary]
            
            # Calculate metrics
            from sklearn.metrics import f1_score, accuracy_score, classification_report
            
            y_true = np.array(true_labels_binary)
            y_pred = np.array(all_predictions)
            
            f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
            f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
            strict_accuracy = accuracy_score(y_true, y_pred)
            
            print(f"\n{'='*70}")
            print("EVALUATION METRICS (on validation set)")
            print(f"{'='*70}")
            print(f"F1 Micro: {f1_micro:.4f}")
            print(f"F1 Macro: {f1_macro:.4f}")
            print(f"Strict Accuracy: {strict_accuracy:.4f}")
            print(f"{'='*70}")
            
            # Print detailed classification report
            print("\nDetailed Classification Report:")
            print(classification_report(y_true, y_pred, target_names=label_list, 
                                      digits=4, zero_division=0))
            print("="*70 + "\n")
    
    # Save results
    if output_path is None:
        # Auto-generate output path
        val_dir = os.path.dirname(val_data_path)
        output_path = os.path.join(val_dir, 'val_predictions_covid_bert.csv')
    
    # Create directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    results_df.to_csv(output_path, index=False)
    print(f"✅ Predictions saved to: {output_path}")
    print(f"   Total samples: {len(results_df)}")
    
    return results_df


def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Predict labels using fine-tuned COVID Twitter BERT model")
    parser.add_argument("--model_path", type=str, 
                       default="/home/sohampoddar/Restored Data/HDD2/sohampoddar/utsav/Two_Step/covid_twitter_bert_v2_model",
                       help="Path to saved model directory")
    parser.add_argument("--tokenizer_path", type=str,
                       default="/home/sohampoddar/Restored Data/HDD2/sohampoddar/utsav/Two_Step/Covid Bert/covid_twitter_bert_v2_tokenizer",
                       help="Path to saved tokenizer directory")
    parser.add_argument("--val_data_path", type=str,
                       default="Dataset/val.csv",
                       help="Path to validation CSV file")
    parser.add_argument("--output_path", type=str, default="/home/sohampoddar/Restored Data/HDD2/sohampoddar/utsav/Two_Step/Covid Bert/val_predictions_covid_bert.csv",
                       help="Path to save predictions CSV (auto-generated if not provided)")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Batch size for inference")
    
    args = parser.parse_args()
    
    # Convert relative paths to absolute if needed
    base_dir = Path(__file__).parent.parent.parent
    
    # Resolve paths - handle both absolute and relative paths
    def resolve_path(path, is_absolute):
        if is_absolute:
            return Path(path)
        else:
            return base_dir / path
    
    model_path = resolve_path(args.model_path, os.path.isabs(args.model_path))
    tokenizer_path = resolve_path(args.tokenizer_path, os.path.isabs(args.tokenizer_path))
    val_data_path = resolve_path(args.val_data_path, os.path.isabs(args.val_data_path))
    
    if args.output_path:
        output_path = resolve_path(args.output_path, os.path.isabs(args.output_path))
    else:
        output_path = None
    
    # Convert Path objects to strings
    model_path = str(model_path.resolve())
    tokenizer_path = str(tokenizer_path.resolve())
    val_data_path = str(val_data_path.resolve())
    output_path = str(output_path.resolve()) if output_path else None
    
    # Run prediction
    results = predict_labels(
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        val_data_path=val_data_path,
        output_path=output_path,
        batch_size=args.batch_size
    )
    
    return results


if __name__ == "__main__":
    main()

# python generate_covid_bert.py --model_path "/home/sohampoddar/Restored Data/HDD2/sohampoddar/utsav/Two_Step/covid_twitter_bert_v2_model" --tokenizer_path "/home/sohampoddar/Restored Data/HDD2/sohampoddar/utsav/Two_Step/covid_twitter_bert_v2_tokenizer" --val_data_path "/home/sohampoddar/Restored Data/HDD2/sohampoddar/utsav/Two_Step/Covid Bert/val.csv" --output_path "/home/sohampoddar/Restored Data/HDD2/sohampoddar/utsav/Two_Step/Covid Bert/val_predictions_covid_bert.csv" --batch_size 16