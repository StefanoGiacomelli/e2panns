"""
AudioSet Metadata Parser
========================
Script to generate AudioSet_EV_Positives.csv and AudioSet_EV_Negatives.csv
from AudioSet metadata files and datasets_mapping.json.

Author: Stefano Giacomelli - Ph.D. candidate in ICT (DISIM dpt. - University of L'Aquila)
"""

import os
import json
import csv
import pandas as pd
from collections import Counter
from typing import Dict, Set, List, Tuple


def load_datasets_mapping(json_path: str) -> Dict[str, int]:
    """
    Load AudioSet label mapping from datasets_mapping.json.
    
    Returns:
        Dict mapping display_name to binary label (0 or 1)
    """
    with open(json_path, 'r') as f:
        all_mappings = json.load(f)
    
    return all_mappings.get("AUDIOSET", {})


def load_class_labels(csv_path: str) -> Dict[str, str]:
    """
    Load class labels mapping from class_labels_indices.csv.
    
    Returns:
        Dict mapping display_name to MID
    """
    display_to_mid = {}
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            display_name = row['display_name'].strip('"')
            mid = row['mid']
            display_to_mid[display_name] = mid
    
    return display_to_mid


def identify_positive_negative_mids(label_mapping: Dict[str, int], 
                                    display_to_mid: Dict[str, str]) -> Tuple[Set[str], Set[str]]:
    """
    Identify positive and negative MIDs based on label mapping.
    
    Args:
        label_mapping: Dict mapping display_name to label (0 or 1)
        display_to_mid: Dict mapping display_name to MID
    
    Returns:
        Tuple of (positive_mids, negative_mids)
    """
    positive_mids = set()
    negative_mids = set()
    
    missing_displays = []
    
    for display_name, label in label_mapping.items():
        if display_name in display_to_mid:
            mid = display_to_mid[display_name]
            if label == 1:
                positive_mids.add(mid)
            elif label == 0:
                negative_mids.add(mid)
        else:
            missing_displays.append(display_name)
    
    if missing_displays:
        print(f"\nWarning: {len(missing_displays)} display names not found in class_labels_indices.csv:")
        for name in missing_displays:
            print(f"  - {name}")
    
    return positive_mids, negative_mids


def parse_positive_labels(labels_str: str) -> List[str]:
    """
    Parse positive_labels string from CSV.
    
    Input: "/m/09x0r,/t/dd00088"
    Output: ['/m/09x0r', '/t/dd00088']
    """
    # Remove quotes and spaces
    labels_str = labels_str.strip().strip('"').strip()
    
    # Split by comma
    if not labels_str:
        return []
    
    mids = [mid.strip() for mid in labels_str.split(',')]
    return mids


def classify_sample(mids: List[str], 
                   positive_mids: Set[str], 
                   negative_mids: Set[str]) -> str:
    """
    Classify a sample as 'positive', 'negative', or 'skip'.
    
    Logic:
    - If contains ANY positive MID → 'positive'
    - If contains ONLY negative MIDs (and NO positive) → 'negative'
    - If contains MIDs not in mapping (and NO positive) → 'skip'
    
    Args:
        mids: List of MIDs in the sample
        positive_mids: Set of positive MIDs
        negative_mids: Set of negative MIDs
    
    Returns:
        'positive', 'negative', or 'skip'
    """
    sample_mids = set(mids)
    
    # Check for positive MIDs
    if sample_mids & positive_mids:
        return 'positive'
    
    # Check if ALL MIDs are in negative set
    relevant_mids = sample_mids & (positive_mids | negative_mids)
    
    if relevant_mids and relevant_mids.issubset(negative_mids):
        return 'negative'
    
    # Sample has MIDs not in our mapping (and no positive MIDs)
    return 'skip'


def read_segments_csv(csv_path: str, 
                     segment_type: str,
                     positive_mids: Set[str],
                     negative_mids: Set[str]) -> Tuple[List[dict], List[dict], Counter]:
    """
    Read a segments CSV and classify samples.
    
    Returns:
        Tuple of (positives_list, negatives_list, stats_counter)
    """
    positives = []
    negatives = []
    stats = Counter()
    
    print(f"\nProcessing {segment_type} segments from: {os.path.basename(csv_path)}")
    
    with open(csv_path, 'r') as f:
        # Skip comment lines
        lines = []
        for line in f:
            if not line.startswith('#'):
                lines.append(line)
        
        # Parse CSV
        reader = csv.DictReader(lines, fieldnames=['YTID', 'start_seconds', 'end_seconds', 'positive_labels'],
                               skipinitialspace=True)
        
        for row in reader:
            ytid = row['YTID'].strip()
            start = float(row['start_seconds'])
            end = float(row['end_seconds'])
            labels_str = row['positive_labels']
            
            # Parse MIDs
            mids = parse_positive_labels(labels_str)
            
            # Classify
            classification = classify_sample(mids, positive_mids, negative_mids)
            
            stats[classification] += 1
            
            if classification == 'positive':
                positives.append({
                    'yt_id': ytid,
                    'start_seconds': start,
                    'end_seconds': end,
                    'positive_labels': str(mids),  # Format as string list
                    'segment_type': segment_type,
                    'downloaded': False
                })
            elif classification == 'negative':
                negatives.append({
                    'yt_id': ytid,
                    'start_seconds': start,
                    'end_seconds': end,
                    'positive_labels': str(mids),
                    'segment_type': segment_type,
                    'downloaded': False
                })
    
    print(f"  - Positives: {stats['positive']}")
    print(f"  - Negatives: {stats['negative']}")
    print(f"  - Skipped: {stats['skip']}")
    
    return positives, negatives, stats


def save_csv(data: List[dict], output_path: str):
    """Save data to CSV file."""
    if not data:
        print(f"Warning: No data to save to {output_path}")
        return
    
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"\nSaved {len(data)} samples to: {output_path}")


def print_label_statistics(positives: List[dict], 
                          negatives: List[dict],
                          positive_mids: Set[str],
                          negative_mids: Set[str],
                          display_to_mid: Dict[str, str]):
    """Print statistics about label distribution."""
    
    # Reverse mapping: MID → display_name
    mid_to_display = {mid: name for name, mid in display_to_mid.items()}
    
    print("\n" + "=" * 80)
    print("LABEL STATISTICS")
    print("=" * 80)
    
    # Count positive labels
    print("\n--- POSITIVE LABELS ---")
    pos_label_counts = Counter()
    for sample in positives:
        mids = eval(sample['positive_labels'])  # Convert string back to list
        for mid in mids:
            if mid in positive_mids:
                pos_label_counts[mid] += 1
    
    for mid, count in pos_label_counts.most_common():
        display = mid_to_display.get(mid, mid)
        print(f"  {display} ({mid}): {count} samples")
    
    # Count negative labels
    print("\n--- NEGATIVE LABELS ---")
    neg_label_counts = Counter()
    for sample in negatives:
        mids = eval(sample['positive_labels'])
        for mid in mids:
            if mid in negative_mids:
                neg_label_counts[mid] += 1
    
    for mid, count in neg_label_counts.most_common():
        display = mid_to_display.get(mid, mid)
        print(f"  {display} ({mid}): {count} samples")


def main():
    """Main execution function."""
    
    print("=" * 80)
    print("AUDIOSET METADATA PARSER")
    print("=" * 80)
    
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    metadata_dir = script_dir
    datasets_root = os.path.dirname(os.path.dirname(script_dir))
    mapping_json = os.path.join(datasets_root, "datasets_mapping.json")
    
    class_labels_csv = os.path.join(metadata_dir, "class_labels_indices.csv")
    balanced_train_csv = os.path.join(metadata_dir, "balanced_train_segments.csv")
    eval_csv = os.path.join(metadata_dir, "eval_segments.csv")
    unbalanced_train_csv = os.path.join(metadata_dir, "unbalanced_train_segments.csv")
    
    output_dir = os.path.dirname(metadata_dir)
    positives_output = os.path.join(output_dir, "EV_Positives.csv")
    negatives_output = os.path.join(output_dir, "EV_Negatives.csv")
    
    # Step 1: Load label mapping
    print("\n[1/5] Loading datasets_mapping.json...")
    label_mapping = load_datasets_mapping(mapping_json)
    print(f"  - Loaded {len(label_mapping)} labels from AUDIOSET mapping")
    print(f"  - Positive labels: {sum(1 for v in label_mapping.values() if v == 1)}")
    print(f"  - Negative labels: {sum(1 for v in label_mapping.values() if v == 0)}")
    
    # Step 2: Load class labels
    print("\n[2/5] Loading class_labels_indices.csv...")
    display_to_mid = load_class_labels(class_labels_csv)
    print(f"  - Loaded {len(display_to_mid)} class labels")
    
    # Step 3: Identify positive and negative MIDs
    print("\n[3/5] Identifying positive and negative MIDs...")
    positive_mids, negative_mids = identify_positive_negative_mids(label_mapping, display_to_mid)
    print(f"  - Positive MIDs: {len(positive_mids)}")
    print(f"    {positive_mids}")
    print(f"  - Negative MIDs: {len(negative_mids)}")
    
    # Step 4: Process all segment CSV files
    print("\n[4/5] Processing segment CSV files...")
    
    all_positives = []
    all_negatives = []
    total_stats = Counter()
    
    # Process balanced train
    if os.path.exists(balanced_train_csv):
        pos, neg, stats = read_segments_csv(balanced_train_csv, 'balanced_train', positive_mids, negative_mids)
        all_positives.extend(pos)
        all_negatives.extend(neg)
        total_stats.update(stats)
    
    # Process eval
    if os.path.exists(eval_csv):
        pos, neg, stats = read_segments_csv(eval_csv, 'eval', positive_mids, negative_mids)
        all_positives.extend(pos)
        all_negatives.extend(neg)
        total_stats.update(stats)
    
    # Process unbalanced train
    if os.path.exists(unbalanced_train_csv):
        pos, neg, stats = read_segments_csv(unbalanced_train_csv, 'unbalanced_train', positive_mids, negative_mids)
        all_positives.extend(pos)
        all_negatives.extend(neg)
        total_stats.update(stats)
    
    # Step 5: Save output CSVs
    print("\n[5/5] Saving output CSV files...")
    save_csv(all_positives, positives_output)
    save_csv(all_negatives, negatives_output)
    
    # Print final statistics
    print("\n" + "=" * 80)
    print("FINAL STATISTICS")
    print("=" * 80)
    print(f"Total samples processed: {sum(total_stats.values())}")
    print(f"  - Positives: {total_stats['positive']}")
    print(f"  - Negatives: {total_stats['negative']}")
    print(f"  - Skipped: {total_stats['skip']}")
    
    # Print label statistics
    print_label_statistics(all_positives, all_negatives, positive_mids, negative_mids, display_to_mid)
    
    print("\n" + "=" * 80)
    print("PARSING COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
