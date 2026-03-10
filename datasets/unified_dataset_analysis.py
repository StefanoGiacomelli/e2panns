"""
Analisi Distribuzione Dataset per Unified Training
===================================================
Questo script analizza la distribuzione di campioni positivi e negativi
nei dataset esistenti per determinare la strategia ottimale di integrazione
di KineScaper-EV.

Strategia proposta:
1. Calcolare totale pos/neg dei 7 dataset esistenti
2. Determinare quanti campioni da KineScaper-EV servono per:
   - RADDOPPIARE i positivi complessivi
   - BILANCIARE perfettamente i negativi post-raddoppio
3. Verificare disponibilità in KineScaper-EV

Author: Stefano Giacomelli
"""

import os
import json
import pandas as pd
from pathlib import Path


# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_ROOT = "./datasets"
KINESCAPER_ROOT = "/mnt/ssd/Kinescaper_EV/dataset/"

# Dataset da analizzare (escludendo KineScaper-EV)
EXISTING_DATASETS = [
    'AudioSet_EV_v1_2025',
    'AudioSet_EV_v2PANNs_2020',
    'sireNNet',
    'LSSiren',
    'ESC50',
    'FSD50K',
    'UrbanSound8K'
]


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def load_datasets_mapping():
    """Load datasets_mapping.json for label mappings."""
    mapping_path = os.path.join(DATA_ROOT, "datasets_mapping.json")
    with open(mapping_path, 'r') as f:
        return json.load(f)


def count_audioset_ev_v1():
    """Count AudioSet_EV_v1_2025 samples."""
    dataset_path = os.path.join(DATA_ROOT, 'AudioSet_EV_v1_2025')
    
    pos_csv = os.path.join(dataset_path, "EV_Positives.csv")
    neg_csv = os.path.join(dataset_path, "EV_Negatives.csv")
    
    pos_df = pd.read_csv(pos_csv)
    neg_df = pd.read_csv(neg_csv)
    
    # Count only downloaded samples
    pos_count = len(pos_df[pos_df['downloaded'] == True])
    neg_count = len(neg_df[neg_df['downloaded'] == True])
    
    return pos_count, neg_count


def count_audioset_ev_v2():
    """Count AudioSet_EV_v2PANNs_2020 samples."""
    dataset_path = os.path.join(DATA_ROOT, 'AudioSet_EV_v2PANNs_2020')
    
    pos_csv = os.path.join(dataset_path, "EV_Positives.csv")
    neg_csv = os.path.join(dataset_path, "EV_Negatives.csv")
    
    pos_df = pd.read_csv(pos_csv)
    neg_df = pd.read_csv(neg_csv)
    
    pos_count = len(pos_df[pos_df['downloaded'] == True])
    neg_count = len(neg_df[neg_df['downloaded'] == True])
    
    return pos_count, neg_count


def count_sirennet():
    """Count sireNNet samples."""
    dataset_path = os.path.join(DATA_ROOT, 'sireNNet')
    
    if not os.path.exists(dataset_path):
        return 0, 0
    
    # Positive classes: ambulance, firetruck, police (from datasets_mapping.json)
    # Negative class: traffic
    pos_count = 0
    neg_count = 0
    
    # Count positives
    for category in ['ambulance', 'firetruck', 'police']:
        category_path = os.path.join(dataset_path, category)
        if os.path.exists(category_path):
            wav_files = [f for f in os.listdir(category_path) if f.endswith('.wav')]
            pos_count += len(wav_files)
    
    # Count negatives
    traffic_path = os.path.join(dataset_path, 'traffic')
    if os.path.exists(traffic_path):
        wav_files = [f for f in os.listdir(traffic_path) if f.endswith('.wav')]
        neg_count = len(wav_files)
    
    return pos_count, neg_count


def count_lssiren():
    """Count LSSiren samples."""
    dataset_path = os.path.join(DATA_ROOT, 'LSSiren')
    
    if not os.path.exists(dataset_path):
        return 0, 0
    
    # Ambulance CSV = positives, Road CSV = negatives
    ambulance_csv = os.path.join(dataset_path, 'Ambulance_final.csv')
    road_csv = os.path.join(dataset_path, 'Road_final.csv')
    
    pos_count = 0
    neg_count = 0
    
    if os.path.exists(ambulance_csv):
        df = pd.read_csv(ambulance_csv)
        pos_count = len(df)
    
    if os.path.exists(road_csv):
        df = pd.read_csv(road_csv)
        neg_count = len(df)
    
    return pos_count, neg_count


def count_esc50():
    """Count ESC50 samples (only siren class as positive)."""
    dataset_path = os.path.join(DATA_ROOT, 'ESC50')
    csv_path = os.path.join(dataset_path, 'esc50.csv')
    
    if not os.path.exists(csv_path):
        return 0, 0
    
    df = pd.read_csv(csv_path)
    
    # Load label mapping from datasets_mapping.json
    mapping_path = os.path.join(DATA_ROOT, 'datasets_mapping.json')
    with open(mapping_path, 'r') as f:
        label_map = json.load(f)['ESC50']
    
    # Positive: categories with label=1 (siren)
    # Negative: categories with label=0 (helicopter, chainsaw, car_horn, etc.)
    positive_categories = [cat for cat, label in label_map.items() if label == 1]
    negative_categories = [cat for cat, label in label_map.items() if label == 0]
    
    pos_count = len(df[df['category'].isin(positive_categories)])
    neg_count = len(df[df['category'].isin(negative_categories)])
    
    return pos_count, neg_count


def count_fsd50k():
    """Count FSD50K samples."""
    dataset_path = os.path.join(DATA_ROOT, 'FSD50K')
    
    # Load mapping to identify siren classes
    mapping = load_datasets_mapping()
    positive_labels = [label for label, val in mapping["FSD50K"].items() if val == 1]
    
    # Check dev and eval sets
    dev_csv = os.path.join(dataset_path, 'FSD50K.ground_truth', 'dev.csv')
    eval_csv = os.path.join(dataset_path, 'FSD50K.ground_truth', 'eval.csv')
    
    pos_count = 0
    neg_count = 0
    
    for csv_path in [dev_csv, eval_csv]:
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                labels = row['labels'].split(',')
                is_positive = any(label in positive_labels for label in labels)
                if is_positive:
                    pos_count += 1
                else:
                    neg_count += 1
    
    return pos_count, neg_count


def count_urbansound8k():
    """Count UrbanSound8K samples."""
    dataset_path = os.path.join(DATA_ROOT, 'UrbanSound8K')
    metadata_path = os.path.join(dataset_path, 'metadata', 'UrbanSound8K.csv')
    
    if not os.path.exists(metadata_path):
        return 0, 0
    
    df = pd.read_csv(metadata_path)
    
    # Class 7 = siren (positive), all others = negative
    pos_count = len(df[df['classID'] == 7])
    neg_count = len(df[df['classID'] != 7])
    
    return pos_count, neg_count


def count_kinescaper_ev():
    """Count KineScaper-EV available samples."""
    metadata_path = os.path.join(KINESCAPER_ROOT, 'json', 'metadata.json')
    
    if not os.path.exists(metadata_path):
        print(f"⚠️  Warning: KineScaper metadata not found at {metadata_path}")
        return 0, 0
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Count positive chunks (10s chunks with overlap >= 0.5s)
    pos_count = 0
    for entry in metadata['dataset_metadata']:
        onset = entry['onset']
        offset = entry['offset']
        
        # Check 4 chunks (0-10s, 10-20s, 20-30s, 30-40s)
        for chunk_idx in range(4):
            chunk_start = chunk_idx * 10.0
            chunk_end = (chunk_idx + 1) * 10.0
            
            # Calculate overlap
            overlap_start = max(chunk_start, onset)
            overlap_end = min(chunk_end, offset)
            overlap_duration = max(0, overlap_end - overlap_start)
            
            if overlap_duration >= 0.5:
                pos_count += 1
    
    # Negatives: from Negatives folder (10 files, augmented 10x)
    negatives_dir = os.path.join(DATA_ROOT, 'KineScaper_EV', 'Negatives')
    if not os.path.exists(negatives_dir):
        negatives_dir = os.path.join(KINESCAPER_ROOT, 'Negatives')
    
    if os.path.exists(negatives_dir):
        # Approximate: 10 files × ~122 base chunks × 10x augmentation = ~12,200
        neg_count = 12180  # From our analysis
    else:
        neg_count = 0
        print(f"⚠️  Warning: Negatives folder not found")
    
    return pos_count, neg_count


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def main():
    print("=" * 80)
    print("ANALISI DISTRIBUZIONE DATASET PER UNIFIED TRAINING")
    print("=" * 80)
    print()
    
    # Collect counts from existing datasets
    dataset_counts = {}
    
    print("Analisi dataset esistenti:")
    print("-" * 80)
    
    for dataset_name in EXISTING_DATASETS:
        print(f"  Analyzing {dataset_name}...", end=" ")
        
        if dataset_name == 'AudioSet_EV_v1_2025':
            pos, neg = count_audioset_ev_v1()
        elif dataset_name == 'AudioSet_EV_v2PANNs_2020':
            pos, neg = count_audioset_ev_v2()
        elif dataset_name == 'sireNNet':
            pos, neg = count_sirennet()
        elif dataset_name == 'LSSiren':
            pos, neg = count_lssiren()
        elif dataset_name == 'ESC50':
            pos, neg = count_esc50()
        elif dataset_name == 'FSD50K':
            pos, neg = count_fsd50k()
        elif dataset_name == 'UrbanSound8K':
            pos, neg = count_urbansound8k()
        else:
            pos, neg = 0, 0
        
        dataset_counts[dataset_name] = {'pos': pos, 'neg': neg, 'total': pos + neg}
        print(f"{pos:>6,} pos + {neg:>6,} neg = {pos+neg:>7,} total")
    
    print()
    
    # Calculate totals
    total_pos = sum(d['pos'] for d in dataset_counts.values())
    total_neg = sum(d['neg'] for d in dataset_counts.values())
    total_samples = total_pos + total_neg
    
    print("=" * 80)
    print("TOTALI DATASET ESISTENTI (7 dataset):")
    print("=" * 80)
    print(f"  Positivi:  {total_pos:>7,} ({total_pos/total_samples*100:>5.1f}%)")
    print(f"  Negativi:  {total_neg:>7,} ({total_neg/total_samples*100:>5.1f}%)")
    print(f"  Totale:    {total_samples:>7,}")
    print(f"  Ratio pos:neg = {total_pos/total_neg if total_neg > 0 else float('inf'):.2f}:1")
    print()
    
    # Analyze KineScaper-EV availability
    print("=" * 80)
    print("KINESCAPER-EV DISPONIBILITÀ:")
    print("=" * 80)
    kinescaper_pos, kinescaper_neg = count_kinescaper_ev()
    print(f"  Positivi disponibili:  {kinescaper_pos:>7,}")
    print(f"  Negativi disponibili:  {kinescaper_neg:>7,}")
    print(f"  Totale disponibile:    {kinescaper_pos + kinescaper_neg:>7,}")
    print()
    
    # Calculate strategy (OPZIONE B MODIFICATA as per user request)
    print("=" * 80)
    print("STRATEGIA DI INTEGRAZIONE (OPZIONE B MODIFICATA):")
    print("=" * 80)
    print()
    print("Strategia:")
    print("  1. NEGATIVI: Prendere TUTTI i negativi disponibili")
    print(f"     - Da altri 7 dataset: {total_neg:,}")
    print(f"     - Da KineScaper-EV (TUTTI): {kinescaper_neg:,}")
    total_negatives_final = total_neg + kinescaper_neg
    print(f"     - Totale negativi finali: {total_negatives_final:,}")
    print()
    print("  2. POSITIVI: Bilanciare 50/50 con i negativi")
    print(f"     - Da altri 7 dataset: {total_pos:,}")
    print(f"     - Target totale positivi (50/50): {total_negatives_final:,}")
    needed_pos_from_kine = total_negatives_final - total_pos
    print(f"     - Servono da KineScaper-EV: {needed_pos_from_kine:,}")
    print()
    print("  3. RE-SAMPLING: Campionare positivi diversi da KineScaper ogni epoca")
    print(f"     - Positivi KineScaper disponibili: {kinescaper_pos:,}")
    print(f"     - Positivi KineScaper necessari/epoca: {needed_pos_from_kine:,}")
    coverage_per_epoch = (needed_pos_from_kine / kinescaper_pos) * 100 if kinescaper_pos > 0 else 0
    print(f"     - Copertura per epoca: {coverage_per_epoch:.1f}%")
    print(f"     - Con re-sampling, si aumenta la diversità vista nel training")
    print()
    
    # Check feasibility
    print("=" * 80)
    print("VERIFICA FATTIBILITÀ:")
    print("=" * 80)
    print()
    print("Negativi:")
    print(f"  Totali richiesti: {total_negatives_final:,} ✓ (usando TUTTI i disponibili)")
    print()
    print("Positivi da KineScaper-EV:")
    print(f"  Richiesti per epoca: {needed_pos_from_kine:,}")
    print(f"  Disponibili totali:  {kinescaper_pos:,}", end="")
    
    feasible_pos = kinescaper_pos >= needed_pos_from_kine
    if feasible_pos:
        print(" ✓ FATTIBILE")
        print(f"  Utilizzo per epoca:  {coverage_per_epoch:.1f}%")
    else:
        print(" ✗ INSUFFICIENTI!")
        print(f"  Mancano: {needed_pos_from_kine - kinescaper_pos:,} positivi")
    
    # Calculate epochs for full coverage
    epochs_for_full_coverage = kinescaper_pos / needed_pos_from_kine if needed_pos_from_kine > 0 else 0
    print()
    print(f"Epoche per vedere TUTTI i positivi KineScaper: {epochs_for_full_coverage:.1f}")
    if epochs_for_full_coverage > 0:
        print(f"  → Con  50 epoche: ogni positivo visto ~{50/epochs_for_full_coverage:.1f}x in media")
        print(f"  → Con 100 epoche: ogni positivo visto ~{100/epochs_for_full_coverage:.1f}x in media")
    print()
    
    # Final summary
    print("=" * 80)
    print("DATASET UNIFICATO FINALE (per epoca):")
    print("=" * 80)
    
    if feasible_pos:
        final_pos = total_pos + needed_pos_from_kine
        final_neg = total_negatives_final
    else:
        # Use all available KineScaper positives even if not enough
        final_pos = total_pos + kinescaper_pos
        final_neg = total_negatives_final
    
    final_total = final_pos + final_neg
    
    print(f"  Positivi:  {final_pos:>7,} ({final_pos/final_total*100:>5.1f}%)")
    print(f"    - Da altri dataset: {total_pos:,} ({total_pos/final_pos*100:.1f}%)")
    print(f"    - Da KineScaper-EV: {final_pos - total_pos:,} ({(final_pos - total_pos)/final_pos*100:.1f}%)")
    print()
    print(f"  Negativi:  {final_neg:>7,} ({final_neg/final_total*100:>5.1f}%)")
    print(f"    - Da altri dataset: {total_neg:,} ({total_neg/final_neg*100:.1f}%)")
    print(f"    - Da KineScaper-EV: {kinescaper_neg:,} ({kinescaper_neg/final_neg*100:.1f}%)")
    print()
    print(f"  Totale:    {final_total:>7,}")
    print(f"  Ratio pos:neg = {final_pos/final_neg if final_neg > 0 else float('inf'):.4f}:1")
    
    balance_diff = abs(final_pos - final_neg)
    if balance_diff < 10:
        print()
        print("  ✓ Dataset PERFETTAMENTE BILANCIATO 50/50!")
    elif balance_diff / final_total < 0.01:  # less than 1% difference
        print()
        print("  ✓ Dataset molto ben bilanciato!")
    elif final_pos > final_neg:
        print()
        print(f"  ⚠️  Leggermente più positivi che negativi (diff: {balance_diff:,}, {balance_diff/final_total*100:.2f}%)")
    else:
        print()
        print(f"  ⚠️  Leggermente più negativi che positivi (diff: {balance_diff:,}, {balance_diff/final_total*100:.2f}%)")
    
    print()
    print("=" * 80)
    print("BILANCIAMENTO CLASSI DI SIRENA (stratificazione):")
    print("=" * 80)
    print()
    print("KineScaper-EV ha 7 classi di sirene bilanciate:")
    print("  hi-lo, two-tone, wail, phaser, piercer, rumbler, yelp")
    print()
    print(f"Per ogni epoca, campionare {needed_pos_from_kine:,} positivi con:")
    print(f"  - Campionamento stratificato: ~{needed_pos_from_kine//7:,} per classe")
    print(f"  - Re-sampling random epoch-to-epoch mantenendo stratificazione")
    
    print()
    print("=" * 80)
    
    # Save results to JSON for programmatic access
    results = {
        'existing_datasets': dataset_counts,
        'existing_totals': {
            'positives': total_pos,
            'negatives': total_neg,
            'total': total_samples
        },
        'kinescaper_available': {
            'positives': kinescaper_pos,
            'negatives': kinescaper_neg,
            'total': kinescaper_pos + kinescaper_neg
        },
        'strategy': {
            'total_negatives': total_negatives_final,
            'needed_positives_from_kinescaper': needed_pos_from_kine,
            'epochs_for_full_coverage': epochs_for_full_coverage,
            'coverage_per_epoch_percent': coverage_per_epoch,
            'feasible': feasible_pos,
            'uses_all_negatives': True,
            'uses_resampling': True
        },
        'unified_final': {
            'positives': final_pos,
            'positives_from_others': total_pos,
            'positives_from_kinescaper': final_pos - total_pos,
            'negatives': final_neg,
            'negatives_from_others': total_neg,
            'negatives_from_kinescaper': kinescaper_neg,
            'total': final_total,
            'balance_ratio': final_pos / final_neg if final_neg > 0 else None,
            'balance_difference': abs(final_pos - final_neg),
            'is_balanced': balance_diff / final_total < 0.01 if final_total > 0 else False
        }
    }
    
    output_file = 'dataset_distribution_analysis.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Risultati salvati in: {output_file}")
    print()


if __name__ == "__main__":
    main()
