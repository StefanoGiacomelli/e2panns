"""
SALT Label Mapping Script
==========================
This script uses py-salt to map standard emergency vehicle labels 
to dataset-specific labels across multiple audio datasets.

Datasets supported: AudioSet, FSD50K, UrbanSound8K (US8K), ESC-50

Strategy: 
- Primary search for 'emergency_vehicle' and 'vehicle_siren_ringing'
- If datasets have NULL in both primary searches, add 'siren_ringing' mapping
"""

import json
import os
from py_salt import EventExplorer


# Standard labels to map (primary search)
PRIMARY_LABELS = ["emergency_vehicle", "vehicle_siren_ringing"]

# Fallback label for datasets without specific emergency vehicle mappings
FALLBACK_LABEL = "siren_ringing"

# Dataset names: SALT name -> Our JSON name
DATASET_MAPPING = {"AudioSet": "AudioSet", "Freesound50k": "FSD50K", "UrbanSound8k": "US8K", "Esc50": "ESC50"}


def print_separator(char="=", length=80):
    """Print a separator line."""
    print(char * length)


def print_mapping_results(dataset_our_name, labels):
    """Print mapping results for a single dataset."""
    if labels:
        print(f"  ✓ {dataset_our_name:12s}: {labels}")
        return True
    else:
        print(f"  ✗ {dataset_our_name:12s}: NULL")
        return False


def print_label_header(std_label):
    """Print header for a standard label."""
    print(f"\n{'─' * 80}")
    print(f"Standard Label: '{std_label}'")
    print(f"{'─' * 80}")


def create_label_mapping_json(explorer, output_path):
    """
    Create label mapping JSON from SALT mappings with conditional fallback.
    
    Strategy:
    1. Search for primary labels (emergency_vehicle, vehicle_siren_ringing)
    2. Check if any dataset has NULL in BOTH primary labels
    3. If yes, add siren_ringing mapping for ALL datasets
    
    Args:
        explorer: EventExplorer instance
        output_path: Path where to save the JSON file
    """
    label_mapping = {}
    datasets_need_fallback = set()
    
    print_separator("=")
    print("SALT POSITIVE LABEL MAPPING - Emergency Vehicle Sirens")
    print("Strategy: Primary labels with conditional 'siren_ringing' fallback")
    print_separator("=")
    
    # Step 1: Search for primary labels
    for std_label in PRIMARY_LABELS:
        try:
            # Get primary mapping from SALT
            primary_mapping = explorer.get_mapping_for_std_label(std_label)
            
            # Print header
            print_label_header(std_label)
            
            if not primary_mapping:
                print(f"  ⚠️  WARNING: No mappings found for '{std_label}' in SALT")
            
            # Build JSON structure for primary label
            label_mapping[std_label] = {}
            
            for dataset_salt_name, dataset_our_name in DATASET_MAPPING.items():
                # Check if mapping exists
                if dataset_salt_name in primary_mapping and primary_mapping[dataset_salt_name]:
                    label_mapping[std_label][dataset_our_name] = primary_mapping[dataset_salt_name]
                    print_mapping_results(dataset_our_name, primary_mapping[dataset_salt_name])
                else:
                    # No mapping found - set to null
                    label_mapping[std_label][dataset_our_name] = None
                    print_mapping_results(dataset_our_name, None)
                    datasets_need_fallback.add(dataset_our_name)
        
        except Exception as e:
            print(f"\n  ❌ ERROR processing '{std_label}': {e}")
            label_mapping[std_label] = {"AudioSet": None, "FSD50K": None, "US8K": None, "ESC50": None}
            datasets_need_fallback.update(DATASET_MAPPING.values())
    
    # Step 2: Check if any dataset has NULL in BOTH primary labels
    datasets_with_both_null = set()
    for dataset_our_name in DATASET_MAPPING.values():
        has_null_in_both = all(label_mapping[primary_label].get(dataset_our_name) is None for primary_label in PRIMARY_LABELS if primary_label in label_mapping)
        if has_null_in_both:
            datasets_with_both_null.add(dataset_our_name)
    
    # Step 3: If needed, add siren_ringing mapping for ALL datasets
    if datasets_with_both_null:
        print(f"\n{'─' * 80}")
        print(f"FALLBACK: Adding '{FALLBACK_LABEL}' mapping")
        print(f"Reason: {len(datasets_with_both_null)} dataset(s) have NULL in both primary labels")
        print(f"Datasets: {', '.join(sorted(datasets_with_both_null))}")
        print(f"{'─' * 80}")
        
        try:
            fallback_mapping = explorer.get_mapping_for_std_label(FALLBACK_LABEL)
            
            print_label_header(FALLBACK_LABEL)
            
            label_mapping[FALLBACK_LABEL] = {}
            
            for dataset_salt_name, dataset_our_name in DATASET_MAPPING.items():
                if dataset_salt_name in fallback_mapping and fallback_mapping[dataset_salt_name]:
                    label_mapping[FALLBACK_LABEL][dataset_our_name] = fallback_mapping[dataset_salt_name]
                    print_mapping_results(dataset_our_name, fallback_mapping[dataset_salt_name])
                else:
                    label_mapping[FALLBACK_LABEL][dataset_our_name] = None
                    print_mapping_results(dataset_our_name, None)
        
        except Exception as e:
            print(f"\n  ❌ ERROR processing fallback '{FALLBACK_LABEL}': {e}")
            label_mapping[FALLBACK_LABEL] = {"AudioSet": None, "FSD50K": None, "US8K": None, "ESC50": None}
    
    # Print summary
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    print(f"Primary labels searched: {', '.join(PRIMARY_LABELS)}")
    if datasets_with_both_null:
        print(f"Fallback label added: {FALLBACK_LABEL}")
        print(f"  Reason: {len(datasets_with_both_null)} dataset(s) need fallback")
    else:
        print("No fallback needed - all datasets have at least one primary mapping")
    print(f"{'=' * 80}\n")
    
    # Save JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(label_mapping, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Label mapping saved to: {output_path}\n")
    
    return label_mapping


def main():
    """Main function to generate positive label mapping."""
    
    # Initialize SALT EventExplorer
    print("Initializing SALT EventExplorer...")
    explorer = EventExplorer()
    print("✓ EventExplorer initialized\n")
    
    # Output path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, "positive_labels_salt_mapping.json")
    
    # Create mapping
    label_mapping = create_label_mapping_json(explorer, output_path)
    
    return label_mapping


if __name__ == "__main__":
    main()
