#!/usr/bin/env python3
"""Analyze Best Models from Benchmark Results
==============================================
Identify the best model+strategy combinations based on:
- Accuracy (highest accuracy)
- Precision (highest precision)
- Recall (highest recall)  
- F1 (highest F1, most balanced)

For both:
- AudioSet_EV_v2PANNs_2020 dataset
- GLOBAL (average across all datasets)

Author: Stefano Giacomelli - Ph.D. candidate in ICT (DISIM dpt. - University of L'Aquila)
"""

import pandas as pd
from pathlib import Path
from typing import Dict, Tuple

# Configuration
PROJECT_ROOT = Path(__file__).parent
FINETUNING_DIR = PROJECT_ROOT / "benchmark_results_finetuning_EV"
RETRAINING_DIR = PROJECT_ROOT / "benchmark_results_re-training_EV"

# Models to analyze (only binary-trained models)
MODELS_FINETUNING = [
    "epanns_finetuned_binary_AS-EV_v1",
    "epanns_finetuned_binary_AS-EV_v2",
    "ced_finetuned_binary_AS-EV_v1",
    "ced_finetuned_binary_AS-EV_v2",
    "clap_finetuned_binary_AS-EV_v1",
    "clap_finetuned_binary_AS-EV_v2",
]

MODELS_RETRAINING = [
    "epanns_retrained_binary_AS-EV_v1",
    "epanns_retrained_binary_AS-EV_v2",
    "ced_retrained_binary_AS-EV_v1",
    "ced_retrained_binary_AS-EV_v2",
]

# Target dataset for specific analysis
TARGET_DATASET = "AudioSet_EV_v2PANNs_2020"

# Metrics to analyze
METRICS = ['Accuracy', 'Precision', 'Recall', 'F1']


def load_benchmark_results(csv_path: Path) -> pd.DataFrame:
    """Load benchmark CSV file."""
    if not csv_path.exists():
        return None
    
    df = pd.read_csv(csv_path)
    
    # Convert numeric columns to float (they might be read as strings)
    numeric_cols = ['Accuracy', 'Precision', 'Recall', 'Specificity', 'F1', 'AUROC', 'FBeta']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df


def extract_model_info(filename: str) -> Tuple[str, str, str]:
    """Extract model name, strategy, and training dataset from filename.
    
    Returns:
        (model_name, strategy, train_dataset)
        e.g., ('epanns', 'finetuned', 'AS-EV_v2')
    """
    parts = filename.replace('.csv', '').split('_')
    
    # Extract model name (first part)
    model_name = parts[0]  # epanns, ced, clap
    
    # Extract strategy
    if 'finetuned' in filename:
        strategy = 'finetuned'
        idx = parts.index('finetuned')
    elif 'retrained' in filename:
        strategy = 'retrained'
        idx = parts.index('retrained')
    else:
        strategy = 'unknown'
        idx = 1
    
    # Extract training dataset (binary, AS-EV_v1 or AS-EV_v2)
    train_task = parts[idx + 1]  # 'binary' or 'multiclass'
    train_dataset = '_'.join(parts[idx + 2:idx + 4])  # 'AS-EV_v1' or 'AS-EV_v2'
    
    return model_name, strategy, train_dataset


def analyze_all_results() -> pd.DataFrame:
    """Analyze all benchmark results and create summary dataframe."""
    
    results = []
    
    # Process finetuning results
    for model_prefix in MODELS_FINETUNING:
        # Find corresponding CSV file
        csv_files = list(FINETUNING_DIR.glob(f"{model_prefix}__binary_results_*.csv"))
        
        if not csv_files:
            print(f"⚠️  No file found for {model_prefix}")
            continue
        
        csv_path = csv_files[0]
        df = load_benchmark_results(csv_path)
        
        if df is None:
            print(f"⚠️  Could not load {csv_path.name}")
            continue
        
        # Extract model info
        model_name, strategy, train_dataset = extract_model_info(csv_path.name)
        
        # Get results for target dataset (AudioSet_EV_v2PANNs_2020)
        target_row = df[df['Dataset'] == TARGET_DATASET]
        
        if not target_row.empty:
            target_metrics = {
                'Model': model_name,
                'Strategy': strategy,
                'Train_Dataset': train_dataset,
                'Eval_Dataset': TARGET_DATASET,
                'Accuracy': float(target_row['Accuracy'].values[0]),
                'Precision': float(target_row['Precision'].values[0]),
                'Recall': float(target_row['Recall'].values[0]),
                'F1': float(target_row['F1'].values[0]),
                'AUROC': float(target_row['AUROC'].values[0]),
                'Specificity': float(target_row['Specificity'].values[0]),
            }
            results.append(target_metrics)
        
        # Compute global average (all datasets)
        global_metrics = {
            'Model': model_name,
            'Strategy': strategy,
            'Train_Dataset': train_dataset,
            'Eval_Dataset': 'GLOBAL',
            'Accuracy': float(df['Accuracy'].mean()),
            'Precision': float(df['Precision'].mean()),
            'Recall': float(df['Recall'].mean()),
            'F1': float(df['F1'].mean()),
            'AUROC': float(df['AUROC'].mean()),
            'Specificity': float(df['Specificity'].mean()),
        }
        results.append(global_metrics)
    
    # Process retraining results (no CLAP)
    for model_prefix in MODELS_RETRAINING:
        csv_files = list(RETRAINING_DIR.glob(f"{model_prefix}__binary_results_*.csv"))
        
        if not csv_files:
            print(f"⚠️  No file found for {model_prefix}")
            continue
        
        csv_path = csv_files[0]
        df = load_benchmark_results(csv_path)
        
        if df is None:
            print(f"⚠️  Could not load {csv_path.name}")
            continue
        
        model_name, strategy, train_dataset = extract_model_info(csv_path.name)
        
        # Get results for target dataset
        target_row = df[df['Dataset'] == TARGET_DATASET]
        
        if not target_row.empty:
            target_metrics = {
                'Model': model_name,
                'Strategy': strategy,
                'Train_Dataset': train_dataset,
                'Eval_Dataset': TARGET_DATASET,
                'Accuracy': float(target_row['Accuracy'].values[0]),
                'Precision': float(target_row['Precision'].values[0]),
                'Recall': float(target_row['Recall'].values[0]),
                'F1': float(target_row['F1'].values[0]),
                'AUROC': float(target_row['AUROC'].values[0]),
                'Specificity': float(target_row['Specificity'].values[0]),
            }
            results.append(target_metrics)
        
        # Global average
        global_metrics = {
            'Model': model_name,
            'Strategy': strategy,
            'Train_Dataset': train_dataset,
            'Eval_Dataset': 'GLOBAL',
            'Accuracy': float(df['Accuracy'].mean()),
            'Precision': float(df['Precision'].mean()),
            'Recall': float(df['Recall'].mean()),
            'F1': float(df['F1'].mean()),
            'AUROC': float(df['AUROC'].mean()),
            'Specificity': float(df['Specificity'].mean()),
        }
        results.append(global_metrics)
    
    return pd.DataFrame(results)


def find_best_models(df: pd.DataFrame) -> Dict:
    """Find best models for each metric and evaluation dataset."""
    
    best_models = {}
    
    for eval_dataset in [TARGET_DATASET, 'GLOBAL']:
        best_models[eval_dataset] = {}
        
        # Filter for this evaluation dataset
        df_subset = df[df['Eval_Dataset'] == eval_dataset].copy()
        
        for metric in METRICS:
            # Find max value
            max_idx = df_subset[metric].idxmax()
            best_row = df_subset.loc[max_idx]
            
            best_models[eval_dataset][metric] = {
                'Model': best_row['Model'],
                'Strategy': best_row['Strategy'],
                'Train_Dataset': best_row['Train_Dataset'],
                'Value': best_row[metric],
                'Full_Name': f"{best_row['Model']}_{best_row['Strategy']}_{best_row['Train_Dataset']}"
            }
    
    return best_models


def print_results(df: pd.DataFrame, best_models: Dict) -> str:
    """Print results in a nice format and return as string."""
    
    output = []
    
    output.append("=" * 100)
    output.append("BENCHMARK RESULTS ANALYSIS - BEST MODELS")
    output.append("=" * 100)
    output.append(f"Analyzed models: {len(df['Model'].unique())} models")
    output.append(f"Strategies: {', '.join(df['Strategy'].unique())}")
    output.append(f"Training datasets: {', '.join(df['Train_Dataset'].unique())}")
    output.append(f"Evaluation: {TARGET_DATASET} + GLOBAL (avg across all datasets)")
    output.append("=" * 100)
    
    # Print full results table
    output.append("\n" + "=" * 100)
    output.append("COMPLETE RESULTS TABLE")
    output.append("=" * 100)
    
    # Sort by Eval_Dataset, then F1
    df_sorted = df.sort_values(['Eval_Dataset', 'F1'], ascending=[True, False])
    
    output.append(df_sorted.to_string(index=False))
    
    # Print best models summary
    output.append("\n" + "=" * 100)
    output.append("BEST MODELS SUMMARY")
    output.append("=" * 100)
    
    for eval_dataset in [TARGET_DATASET, 'GLOBAL']:
        output.append(f"\n{'#' * 100}")
        output.append(f"# {eval_dataset}")
        output.append(f"{'#' * 100}\n")
        
        for metric in METRICS:
            best = best_models[eval_dataset][metric]
            
            # Metric interpretation
            interpretation = {
                'Accuracy': 'Most ACCURATE (overall correctness)',
                'Precision': 'Most PRECISE (few false positives, high confidence when predicting siren)',
                'Recall': 'Most SENSITIVE (few false negatives, catches most sirens)',
                'F1': 'Most BALANCED (best trade-off precision-recall)'
            }
            
            output.append(f"🏆 {metric.upper()}: {best['Value']:.4f}")
            output.append(f"   Model: {best['Model'].upper()}")
            output.append(f"   Strategy: {best['Strategy'].upper()}")
            output.append(f"   Trained on: {best['Train_Dataset']}")
            output.append(f"   → {interpretation[metric]}")
            output.append("")
    
    # Summary statistics
    output.append("\n" + "=" * 100)
    output.append("SUMMARY STATISTICS")
    output.append("=" * 100)
    
    # Best model by number of wins
    wins = {}
    for eval_dataset in [TARGET_DATASET, 'GLOBAL']:
        for metric in METRICS:
            model_name = best_models[eval_dataset][metric]['Full_Name']
            wins[model_name] = wins.get(model_name, 0) + 1
    
    output.append("\nNumber of 'wins' (best metric) per model:")
    for model, count in sorted(wins.items(), key=lambda x: x[1], reverse=True):
        output.append(f"  {model}: {count} wins")
    
    # Overall champion (most wins)
    champion = max(wins.items(), key=lambda x: x[1])
    output.append(f"\n🎖️  OVERALL CHAMPION: {champion[0]} ({champion[1]} wins out of {len(METRICS) * 2} categories)")
    
    # Join all output lines and print
    output_str = "\n".join(output)
    print(output_str)
    
    return output_str


def save_results(df: pd.DataFrame, output_text: str):
    """Save results to CSV and log file."""
    
    output_dir = PROJECT_ROOT / "results" / "benchmarks_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save full results CSV
    csv_path = output_dir / "all_binary_models_comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n✅ Full results saved to: {csv_path}")
    
    # Save complete output log
    log_path = output_dir / "best_models_summary.txt"
    
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(output_text)
    
    print(f"✅ Complete log saved to: {log_path}")


def main():
    """Main execution."""
    print("\n🔍 Analyzing benchmark results...\n")
    
    # Analyze all results
    df = analyze_all_results()
    
    if df.empty:
        print("❌ No results found!")
        return
    
    # Find best models
    best_models = find_best_models(df)
    
    # Print results and get output string
    output_text = print_results(df, best_models)
    
    # Save results
    save_results(df, output_text)
    
    print("\n✅ Analysis complete!\n")


if __name__ == "__main__":
    main()
