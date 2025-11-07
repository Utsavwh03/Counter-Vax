#!/usr/bin/env python3
"""
Process T5 predictions from text file and create a CSV with predicted labels
that can be used for evaluating llama, gemma, and phi-3 models.
"""

import pandas as pd
import re
from pathlib import Path

# Mapping from T5 label descriptions to standard label names
# These are the key phrases that appear in T5 predictions
T5_LABEL_KEYWORDS = {
    'conspiracy': ['deeper conspiracy', 'conspiracy'],
    'pharma': ['against big pharma', 'big pharma', 'pharmaceutical'],
    'ineffective': ['vaccine is ineffective', 'ineffective', 'useless'],
    'mandatory': ['against mandatory vaccination', 'mandatory vaccination', 'mandatory'],
    'ingredients': ['vaccine ingredients', 'ingredients', 'technology', 'mRNA', 'fetal cells'],
    'country': ['country of origin', 'country'],
    'side-effect': ['side effects', 'deaths', 'side-effect', 'adverse reactions'],
    'rushed': ['untested', 'rushed process', 'rushed', 'not tested properly'],
    'political': ['political side', 'political', 'governments', 'politicians'],
    'unnecessary': ['vaccines are unnecessary', 'unnecessary', 'alternate cures', 'better'],
    'none': ['no specific reason', 'no reason', 'other than'],
    'religious': ['religious reasons', 'religious']
}

def extract_labels_from_pred(pred_text):
    """
    Extract label names from T5 prediction text.
    Handles multiple labels in a single prediction.
    
    Args:
        pred_text (str): The prediction text after "Pred:"
        
    Returns:
        list: List of label names (e.g., ['conspiracy', 'pharma'])
    """
    if not pred_text or not pred_text.strip():
        return ['none']
    
    pred_text_lower = pred_text.lower()
    found_labels = set()
    
    # Check each label type and its keywords
    for label_name, keywords in T5_LABEL_KEYWORDS.items():
        for keyword in keywords:
            if keyword in pred_text_lower:
                found_labels.add(label_name)
                break  # Found this label type, move to next
    
    # Special case: if we found other labels, don't include 'none'
    if found_labels and 'none' in found_labels and len(found_labels) > 1:
        found_labels.remove('none')
    
    # If no labels found, return 'none'
    if not found_labels:
        return ['none']
    
    return sorted(list(found_labels))


def parse_t5_predictions_file(predictions_file):
    """
    Parse T5 predictions from text file.
    
    Args:
        predictions_file (str): Path to T5 predictions text file
        
    Returns:
        list: List of predicted labels (each is a list of label names)
    """
    predicted_labels = []
    
    with open(predictions_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Extract prediction part (after "Pred:")
            if 'Pred:' in line:
                pred_part = line.split('Pred:', 1)[1].strip()
                labels = extract_labels_from_pred(pred_part)
                predicted_labels.append(labels)
            else:
                # If no "Pred:" found, assume no labels
                predicted_labels.append(['none'])
    
    return predicted_labels


def create_csv_with_t5_predictions(val_data_path, t5_predictions_file, output_path):
    """
    Create a CSV file with T5 predicted labels, similar to the COVID BERT version.
    
    Args:
        val_data_path (str): Path to validation CSV file
        t5_predictions_file (str): Path to T5 predictions text file
        output_path (str): Path to save output CSV
    """
    print("="*70)
    print("Processing T5 Predictions")
    print("="*70)
    
    # Load validation data
    print(f"\nLoading validation data from: {val_data_path}")
    val_data = pd.read_csv(val_data_path)
    print(f"✅ Loaded {len(val_data)} samples")
    
    # Parse T5 predictions
    print(f"\nParsing T5 predictions from: {t5_predictions_file}")
    predicted_labels = parse_t5_predictions_file(t5_predictions_file)
    print(f"✅ Parsed {len(predicted_labels)} predictions")
    
    # Verify counts match
    if len(predicted_labels) != len(val_data):
        print(f"\n⚠️  Warning: Prediction count ({len(predicted_labels)}) doesn't match validation data count ({len(val_data)})")
        print(f"   Using first {min(len(predicted_labels), len(val_data))} samples")
        min_len = min(len(predicted_labels), len(val_data))
        predicted_labels = predicted_labels[:min_len]
        val_data = val_data.head(min_len)
    
    # Convert predicted labels to string list format (like "['conspiracy', 'pharma']")
    # Using str() on a list produces the exact format we need: "['conspiracy', 'pharma']"
    predicted_labels_str_list = []
    for labels in predicted_labels:
        if labels:
            # Format as string representation of list: "['label1', 'label2']"
            predicted_labels_str_list.append(str(labels))
        else:
            predicted_labels_str_list.append("['none']")
    
    # Create new dataframe with T5 predicted labels
    result_df = val_data.copy()
    
    # Replace labels column with T5 predictions (in string list format)
    result_df['labels'] = predicted_labels_str_list
    
    # Also keep original labels if they exist (rename to true_labels)
    if 'labels' in val_data.columns:
        original_labels = val_data['labels'].tolist()
        # Convert to string format if they're lists
        original_labels_str = []
        for label in original_labels:
            if isinstance(label, str):
                # Try to parse if it's a string representation of list
                if label.startswith('[') and label.endswith(']'):
                    try:
                        import ast
                        label_list = ast.literal_eval(label)
                        if isinstance(label_list, list):
                            original_labels_str.append(' '.join(label_list))
                        else:
                            original_labels_str.append(label)
                    except:
                        original_labels_str.append(label)
                else:
                    original_labels_str.append(label)
            elif isinstance(label, list):
                original_labels_str.append(' '.join(label))
            else:
                original_labels_str.append(str(label))
        
        result_df['true_labels'] = original_labels_str
    
    # Save to CSV
    print(f"\nSaving results to: {output_path}")
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    result_df.to_csv(output_path, index=False)
    print(f"✅ Saved {len(result_df)} samples to {output_path}")
    
    # Print some statistics
    print(f"\n{'='*70}")
    print("Label Distribution (T5 Predictions)")
    print(f"{'='*70}")
    all_labels = []
    for labels_list in predicted_labels:
        all_labels.extend(labels_list)
    
    from collections import Counter
    label_counts = Counter(all_labels)
    for label, count in label_counts.most_common():
        print(f"  {label}: {count}")
    print(f"{'='*70}\n")
    
    return result_df


def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Process T5 predictions and create CSV for evaluation")
    parser.add_argument("--val_data_path", type=str,
                       default="/home/sohampoddar/Restored Data/HDD2/sohampoddar/utsav/Data/val_CA_with_label_desc.csv",
                       help="Path to validation CSV file")
    parser.add_argument("--t5_predictions_file", type=str,
                       default="/home/sohampoddar/Restored Data/HDD2/sohampoddar/utsav/Two_Step/lora_prediction_flan_t5_large_new.txt",
                       help="Path to T5 predictions text file")
    parser.add_argument("--output_path", type=str,
                       default="/home/sohampoddar/Restored Data/HDD2/sohampoddar/utsav/Data/val_CA_with_label_desc_with_predicted_labels_from_t5.csv",
                       help="Path to save output CSV")
    
    args = parser.parse_args()
    
    # Resolve paths
    val_data_path = str(Path(args.val_data_path).resolve())
    t5_predictions_file = str(Path(args.t5_predictions_file).resolve())
    output_path = str(Path(args.output_path).resolve())
    
    # Process and save
    result_df = create_csv_with_t5_predictions(
        val_data_path=val_data_path,
        t5_predictions_file=t5_predictions_file,
        output_path=output_path
    )
    
    return result_df


if __name__ == "__main__":
    main()

