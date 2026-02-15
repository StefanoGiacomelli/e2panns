"""
AudioSet File Copier
====================
Script to copy audio files from Lexar HD to local folders based on CSV files.
Organized by segment_type (balanced_train, eval, unbalanced).

Handles:
1. PHASE 1: Copy balanced_train + eval to MacBook
2. PHASE 2: Extract and copy unbalanced to Lexar HD (selective extraction)

Author: Stefano Giacomelli - Ph.D. candidate in ICT (DISIM dpt. - University of L'Aquila)
"""

import os
import shutil
import subprocess
import pandas as pd
from typing import Dict, List, Tuple, Set
import time


# =============================================================================
# CONFIGURATION
# =============================================================================

LEXAR_BASE = "/Volumes/Lexar/audioset-qiuqiangkong"
LEXAR_BALANCED = os.path.join(LEXAR_BASE, "balanced_train_segments")
LEXAR_EVAL = os.path.join(LEXAR_BASE, "eval_segments")
LEXAR_UNBALANCED = os.path.join(LEXAR_BASE, "unbalanced")

LOCAL_BASE = "./datasets/AudioSet_EV_v2PANNs_2020"
POSITIVES_CSV = os.path.join(LOCAL_BASE, "EV_Positives.csv")
NEGATIVES_CSV = os.path.join(LOCAL_BASE, "EV_Negatives.csv")

# Output folders (will create subfolders for each segment_type)
POSITIVE_OUTPUT = os.path.join(LOCAL_BASE, "Positive_files")
NEGATIVE_OUTPUT = os.path.join(LOCAL_BASE, "Negative_files")

# Unbalanced output on Lexar HD (more space available)
LEXAR_POSITIVE_UNBALANCED = os.path.join(LEXAR_BASE, "Positive_files_unbalanced")


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_output_folders():
    """Create output folder structure."""
    print("\n" + "=" * 80)
    print("CREATING OUTPUT FOLDERS")
    print("=" * 80)
    
    # Local folders for balanced + eval
    for base in [POSITIVE_OUTPUT, NEGATIVE_OUTPUT]:
        for segment in ["balanced_train", "eval"]:
            folder = os.path.join(base, segment)
            os.makedirs(folder, exist_ok=True)
            print(f"✓ {folder}")
    
    # Lexar folders for unbalanced
    os.makedirs(LEXAR_POSITIVE_UNBALANCED, exist_ok=True)
    print(f"✓ {LEXAR_POSITIVE_UNBALANCED}")


def get_free_space_gb(path: str) -> float:
    """Get free space in GB for a given path."""
    stat = os.statvfs(path)
    free_bytes = stat.f_bavail * stat.f_frsize
    return free_bytes / (1024 ** 3)


def remove_Y_prefix(filename: str) -> str:
    """Remove Y prefix from filename if present."""
    if filename.startswith('Y') and not filename.startswith('Y-'):
        return filename[1:]
    return filename


# =============================================================================
# PHASE 1: COPY BALANCED + EVAL
# =============================================================================

def copy_balanced_eval_files(df: pd.DataFrame, 
                             label_type: str,
                             output_base: str) -> Dict[str, int]:
    """
    Copy files from balanced_train and eval segments.
    Skips files that already exist in destination.
    
    Args:
        df: DataFrame with samples
        label_type: "positive" or "negative"
        output_base: Base output folder
    
    Returns:
        Dict with statistics
    """
    stats = {
        'balanced_train': {'found': 0, 'not_found': 0, 'skipped': 0},
        'eval': {'found': 0, 'not_found': 0, 'skipped': 0}
    }
    
    # Track which rows to update
    downloaded_indices = []
    
    for idx, row in df.iterrows():
        yt_id = row['yt_id']
        segment_type = row['segment_type']
        
        # Skip unbalanced (will be handled in Phase 2)
        if segment_type == 'unbalanced_train':
            continue
        
        # Determine source folder
        if segment_type == 'balanced_train':
            source_folder = LEXAR_BALANCED
            output_folder = os.path.join(output_base, "balanced_train")
        elif segment_type == 'eval':
            source_folder = LEXAR_EVAL
            output_folder = os.path.join(output_base, "eval")
        else:
            continue
        
        # Source and destination file paths
        source_file = os.path.join(source_folder, f"{yt_id}.wav")
        dest_file = os.path.join(output_folder, f"{yt_id}.wav")
        
        # Check if destination file already exists (resume capability)
        if os.path.exists(dest_file):
            stats[segment_type]['skipped'] += 1
            stats[segment_type]['found'] += 1
            # Mark as downloaded in CSV if not already
            if not row['downloaded']:
                downloaded_indices.append(idx)
            continue
        
        # Check if source file exists and copy
        if os.path.exists(source_file):
            try:
                shutil.copy2(source_file, dest_file)
                stats[segment_type]['found'] += 1
                downloaded_indices.append(idx)
            except Exception as e:
                print(f"Error copying {yt_id}: {e}")
                stats[segment_type]['not_found'] += 1
        else:
            stats[segment_type]['not_found'] += 1
    
    # Update DataFrame
    df.loc[downloaded_indices, 'downloaded'] = True
    
    return stats, downloaded_indices


def phase1_copy_balanced_eval():
    """Execute Phase 1: Copy balanced and eval files."""
    print("\n" + "=" * 80)
    print("PHASE 1: COPYING BALANCED + EVAL SEGMENTS")
    print("=" * 80)
    
    # Load CSVs
    print("\nLoading CSV files...")
    df_pos = pd.read_csv(POSITIVES_CSV)
    df_neg = pd.read_csv(NEGATIVES_CSV)
    
    print(f"  - Positives: {len(df_pos)} samples")
    print(f"  - Negatives: {len(df_neg)} samples")
    
    # Copy positives
    print("\n--- COPYING POSITIVES ---")
    stats_pos, indices_pos = copy_balanced_eval_files(df_pos, "positive", POSITIVE_OUTPUT)
    
    print(f"\nBalanced Train:")
    print(f"  ✓ Total: {stats_pos['balanced_train']['found']}")
    print(f"  ↻ Skipped (already exists): {stats_pos['balanced_train']['skipped']}")
    print(f"  ✗ Not found: {stats_pos['balanced_train']['not_found']}")
    
    print(f"\nEval:")
    print(f"  ✓ Total: {stats_pos['eval']['found']}")
    print(f"  ↻ Skipped (already exists): {stats_pos['eval']['skipped']}")
    print(f"  ✗ Not found: {stats_pos['eval']['not_found']}")
    
    # Copy negatives
    print("\n--- COPYING NEGATIVES ---")
    stats_neg, indices_neg = copy_balanced_eval_files(df_neg, "negative", NEGATIVE_OUTPUT)
    
    print(f"\nBalanced Train:")
    print(f"  ✓ Total: {stats_neg['balanced_train']['found']}")
    print(f"  ↻ Skipped (already exists): {stats_neg['balanced_train']['skipped']}")
    print(f"  ✗ Not found: {stats_neg['balanced_train']['not_found']}")
    
    print(f"\nEval:")
    print(f"  ✓ Total: {stats_neg['eval']['found']}")
    print(f"  ↻ Skipped (already exists): {stats_neg['eval']['skipped']}")
    print(f"  ✗ Not found: {stats_neg['eval']['not_found']}")
    
    # Save updated CSVs
    print("\n--- UPDATING CSV FILES ---")
    df_pos.to_csv(POSITIVES_CSV, index=False)
    print(f"✓ Updated {POSITIVES_CSV}")
    df_neg.to_csv(NEGATIVES_CSV, index=False)
    print(f"✓ Updated {NEGATIVES_CSV}")
    
    return df_pos, df_neg, stats_pos, stats_neg


# =============================================================================
# PHASE 2: EXTRACT AND COPY UNBALANCED
# =============================================================================

def analyze_zip_contents(zip_path: str, needed_ytids: Set[str]) -> Tuple[List[str], int]:
    """
    Analyze ZIP contents without extracting using 7z.
    Supports multi-part archives (.z01, .z02, .zip).
    
    Args:
        zip_path: Path to ZIP file
        needed_ytids: Set of YouTube IDs we need
    
    Returns:
        Tuple of (list of matching files, total size in bytes)
    """
    matching_files = []
    total_size = 0
    
    try:
        # Use 7z to list archive contents with details
        result = subprocess.run(
            ['7z', 'l', '-slt', zip_path],
            capture_output=True,
            text=True,
            timeout=120
        )
        
        if result.returncode != 0:
            print(f"Error reading {os.path.basename(zip_path)}: 7z exit code {result.returncode}")
            return matching_files, total_size
        
        # Parse 7z output
        lines = result.stdout.split('\n')
        current_file = None
        current_size = 0
        
        for line in lines:
            if line.startswith('Path = '):
                current_file = line[7:].strip()
            elif line.startswith('Size = '):
                try:
                    current_size = int(line[7:].strip())
                except:
                    current_size = 0
            elif line == '':  # End of file entry
                if current_file and current_file.endswith('.wav'):
                    # Remove Y prefix if present
                    clean_name = remove_Y_prefix(os.path.basename(current_file))
                    yt_id = clean_name.replace('.wav', '')
                    
                    if yt_id in needed_ytids:
                        matching_files.append(current_file)
                        total_size += current_size
                
                current_file = None
                current_size = 0
        
    except subprocess.TimeoutExpired:
        print(f"Timeout reading {os.path.basename(zip_path)}")
    except Exception as e:
        print(f"Error reading {os.path.basename(zip_path)}: {e}")
    
    return matching_files, total_size


def extract_and_copy_unbalanced(df: pd.DataFrame,
                                label_type: str,
                                output_folder: str) -> Dict:
    """
    Extract needed files from unbalanced ZIPs and copy to output folder.
    
    Args:
        df: DataFrame with unbalanced samples
        label_type: "positive" or "negative"
        output_folder: Output folder path
    
    Returns:
        Statistics dict
    """
    # Filter unbalanced samples
    df_unbal = df[df['segment_type'] == 'unbalanced_train'].copy()
    
    if len(df_unbal) == 0:
        return {'found': 0, 'not_found': 0, 'total_size_mb': 0}
    
    print(f"\n{label_type.upper()}: {len(df_unbal)} unbalanced samples to extract")
    
    # Get set of needed YouTube IDs
    needed_ytids = set(df_unbal['yt_id'].values)
    
    # Find all ZIP files
    zip_files = sorted([
        os.path.join(LEXAR_UNBALANCED, f)
        for f in os.listdir(LEXAR_UNBALANCED)
        if f.endswith('.zip')
    ])
    
    print(f"Found {len(zip_files)} ZIP files to scan")
    
    # Analyze all ZIPs
    print("\nScanning ZIP contents (this may take a few minutes)...")
    zip_analysis = {}
    total_size_bytes = 0
    total_files_found = 0
    
    for i, zip_path in enumerate(zip_files, 1):
        zip_name = os.path.basename(zip_path)
        matching_files, size = analyze_zip_contents(zip_path, needed_ytids)
        
        if matching_files:
            zip_analysis[zip_path] = matching_files
            total_size_bytes += size
            total_files_found += len(matching_files)
            print(f"  [{i}/{len(zip_files)}] {zip_name}: {len(matching_files)} files ({size / (1024**2):.1f} MB)")
        else:
            print(f"  [{i}/{len(zip_files)}] {zip_name}: 0 files")
    
    print(f"\nTotal files to extract: {total_files_found}")
    print(f"Total size: {total_size_bytes / (1024**3):.2f} GB")
    
    # Check available space
    free_space_gb = get_free_space_gb(LEXAR_BASE)
    needed_space_gb = total_size_bytes / (1024**3)
    
    print(f"\nAvailable space on Lexar: {free_space_gb:.2f} GB")
    print(f"Needed space: {needed_space_gb:.2f} GB")
    
    if needed_space_gb > free_space_gb * 0.9:  # Leave 10% margin
        print("\n⚠️  WARNING: Not enough space on Lexar HD!")
        print("Aborting unbalanced extraction.")
        return {'found': 0, 'not_found': len(df_unbal), 'total_size_mb': 0}
    
    print("✓ Sufficient space available")
    
    # Extract and copy files
    print("\nExtracting and copying files...")
    downloaded_indices = []
    copied_count = 0
    skipped_count = 0
    
    # Create a temp folder for extraction
    temp_extract_folder = os.path.join(output_folder, "_temp_extract")
    os.makedirs(temp_extract_folder, exist_ok=True)
    
    for zip_path, files_to_extract in zip_analysis.items():
        zip_name = os.path.basename(zip_path)
        print(f"\nProcessing {zip_name} ({len(files_to_extract)} files)...")
        
        # Extract files using 7z (batch extraction is faster)
        try:
            # Create a file list for 7z to extract
            for file_in_zip in files_to_extract:
                # Extract this specific file
                result = subprocess.run(
                    ['7z', 'e', zip_path, file_in_zip, f'-o{temp_extract_folder}', '-y'],
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                
                if result.returncode == 0:
                    # File extracted successfully
                    extracted_filename = os.path.basename(file_in_zip)
                    temp_file_path = os.path.join(temp_extract_folder, extracted_filename)
                    
                    if os.path.exists(temp_file_path):
                        # Get clean filename (remove Y prefix)
                        clean_name = remove_Y_prefix(extracted_filename)
                        yt_id = clean_name.replace('.wav', '')
                        
                        # Check if destination file already exists (resume capability)
                        final_path = os.path.join(output_folder, clean_name)
                        
                        if os.path.exists(final_path):
                            # File already exists, skip and mark as downloaded
                            os.remove(temp_file_path)  # Remove temp file
                            mask = (df['yt_id'] == yt_id) & (df['segment_type'] == 'unbalanced_train')
                            indices = df[mask].index.tolist()
                            downloaded_indices.extend(indices)
                            skipped_count += 1
                        else:
                            # Move to final output folder
                            shutil.move(temp_file_path, final_path)
                            
                            # Mark as downloaded in DataFrame
                            mask = (df['yt_id'] == yt_id) & (df['segment_type'] == 'unbalanced_train')
                            indices = df[mask].index.tolist()
                            downloaded_indices.extend(indices)
                            
                            copied_count += 1
                        
                        if (copied_count + skipped_count) % 100 == 0:
                            print(f"  Progress: {copied_count + skipped_count}/{total_files_found} (copied: {copied_count}, skipped: {skipped_count})")
        
        except subprocess.TimeoutExpired:
            print(f"Timeout processing {zip_name}")
        except Exception as e:
            print(f"Error processing {zip_name}: {e}")
    
    # Cleanup temp folder
    try:
        if os.path.exists(temp_extract_folder):
            shutil.rmtree(temp_extract_folder)
    except:
        pass
    
    # Update DataFrame
    df.loc[downloaded_indices, 'downloaded'] = True
    
    stats = {
        'found': copied_count + skipped_count,
        'copied': copied_count,
        'skipped': skipped_count,
        'not_found': len(df_unbal) - copied_count - skipped_count,
        'total_size_mb': total_size_bytes / (1024**2)
    }
    
    print(f"\n✓ Total: {stats['found']} files")
    print(f"  → Newly copied: {copied_count}")
    print(f"  → Skipped (already exists): {skipped_count}")
    print(f"✗ Not found: {stats['not_found']} files")
    
    return stats


def phase2_extract_unbalanced(df_pos: pd.DataFrame, df_neg: pd.DataFrame):
    """Execute Phase 2: Extract and copy unbalanced files (POSITIVES ONLY)."""
    print("\n" + "=" * 80)
    print("PHASE 2: EXTRACTING UNBALANCED SEGMENTS (POSITIVES ONLY)")
    print("=" * 80)
    
    # Extract positives
    print("\n--- EXTRACTING POSITIVES ---")
    stats_pos = extract_and_copy_unbalanced(df_pos, "positive", LEXAR_POSITIVE_UNBALANCED)
    
    # Skip negatives extraction (not needed - already have enough from balanced+eval)
    print("\n--- SKIPPING NEGATIVES UNBALANCED ---")
    print("  Negatives unbalanced extraction skipped (sufficient samples from balanced+eval)")
    stats_neg = {'found': 0, 'copied': 0, 'skipped': 0, 'not_found': 0, 'total_size_mb': 0}
    
    # Save updated CSVs
    print("\n--- UPDATING CSV FILES ---")
    df_pos.to_csv(POSITIVES_CSV, index=False)
    print(f"✓ Updated {POSITIVES_CSV}")
    df_neg.to_csv(NEGATIVES_CSV, index=False)
    print(f"✓ Updated {NEGATIVES_CSV}")
    
    return stats_pos, stats_neg


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def print_final_summary(stats_pos_phase1, stats_neg_phase1, 
                       stats_pos_phase2, stats_neg_phase2):
    """Print final summary statistics."""
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    
    print("\n--- POSITIVES ---")
    total_pos = (stats_pos_phase1['balanced_train']['found'] + 
                 stats_pos_phase1['eval']['found'] + 
                 stats_pos_phase2['found'])
    total_skipped_pos = (stats_pos_phase1['balanced_train']['skipped'] +
                         stats_pos_phase1['eval']['skipped'] +
                         stats_pos_phase2['skipped'])
    print(f"  Balanced Train: {stats_pos_phase1['balanced_train']['found']} total ({stats_pos_phase1['balanced_train']['skipped']} skipped)")
    print(f"  Eval: {stats_pos_phase1['eval']['found']} total ({stats_pos_phase1['eval']['skipped']} skipped)")
    print(f"  Unbalanced: {stats_pos_phase2['found']} total ({stats_pos_phase2['copied']} newly copied, {stats_pos_phase2['skipped']} skipped)")
    print(f"  TOTAL: {total_pos} files ({total_skipped_pos} already existed)")
    
    print("\n--- NEGATIVES ---")
    total_neg = (stats_neg_phase1['balanced_train']['found'] + 
                 stats_neg_phase1['eval']['found'] + 
                 stats_neg_phase2['found'])
    total_skipped_neg = (stats_neg_phase1['balanced_train']['skipped'] +
                         stats_neg_phase1['eval']['skipped'] +
                         stats_neg_phase2.get('skipped', 0))
    print(f"  Balanced Train: {stats_neg_phase1['balanced_train']['found']} total ({stats_neg_phase1['balanced_train']['skipped']} skipped)")
    print(f"  Eval: {stats_neg_phase1['eval']['found']} total ({stats_neg_phase1['eval']['skipped']} skipped)")
    print(f"  Unbalanced: SKIPPED (not needed - sufficient samples from balanced+eval)")
    print(f"  TOTAL: {total_neg} files ({total_skipped_neg} already existed)")
    
    print("\n--- STORAGE ---")
    total_size_mb = stats_pos_phase2['total_size_mb'] + stats_neg_phase2.get('total_size_mb', 0)
    print(f"  Positives unbalanced extracted: {stats_pos_phase2['total_size_mb'] / 1024:.2f} GB")
    
    print("\n--- NOT FOUND ---")
    total_not_found = (stats_pos_phase1['balanced_train']['not_found'] + 
                       stats_pos_phase1['eval']['not_found'] +
                       stats_pos_phase2['not_found'] +
                       stats_neg_phase1['balanced_train']['not_found'] +
                       stats_neg_phase1['eval']['not_found'])
    print(f"  Total files not found: {total_not_found}")


def main():
    """Main execution function."""
    
    start_time = time.time()
    
    print("=" * 80)
    print("AUDIOSET FILE COPIER")
    print("=" * 80)
    print(f"\nStart time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create output folders
    create_output_folders()
    
    # Phase 1: Copy balanced + eval
    df_pos, df_neg, stats_pos_phase1, stats_neg_phase1 = phase1_copy_balanced_eval()
    
    # Phase 2: Extract unbalanced
    stats_pos_phase2, stats_neg_phase2 = phase2_extract_unbalanced(df_pos, df_neg)
    
    # Print final summary
    print_final_summary(stats_pos_phase1, stats_neg_phase1,
                       stats_pos_phase2, stats_neg_phase2)
    
    elapsed = time.time() - start_time
    print(f"\n{'=' * 80}")
    print(f"COMPLETED in {elapsed / 60:.1f} minutes")
    print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)


if __name__ == "__main__":
    main()
