"""
Multi-Class Emergency Vehicle Classification Utilities
=======================================================
Shared utilities for 4-way balanced emergency vehicle classification.

Classes:
    0 = Negative (traffic/non-emergency)
    1 = Police
    2 = Ambulance
    3 = Fire truck

Author: Stefano Giacomelli - Ph.D. candidate in ICT (DISIM dpt. - University of L'Aquila)
"""

import random
import numpy as np
from typing import Dict, List, Tuple, Optional, Union


class FourWayBalancer:
    """
    Balanced 4-class sampler for emergency vehicle classification.
    
    Strategy:
        1. Count pure samples per class
        2. Determine target_count (min class or specified)
        3. Use multi-label samples to fill gaps (priority to largest gaps)
        4. Undersample over-represented classes
    
    Example:
        >>> balancer = FourWayBalancer(target_mode='auto')
        >>> result = balancer.balance(
        ...     pure_samples={0: [0,1,2], 1: [3,4], 2: [5,6], 3: [7,8,9]},
        ...     multi_samples=[(10, [1,2]), (11, [2,3])],
        ...     seed=42
        ... )
        >>> print(result['target_count'])  # 2 (min class)
    """
    
    def __init__(self, 
                 target_mode: Union[str, int] = 'auto',
                 min_samples_per_class: int = 10):
        """
        Initialize 4-way balancer.
        
        Args:
            target_mode: 'auto' (use minimum class count) or int (fixed count per class)
            min_samples_per_class: Minimum viable samples for a class (warning threshold)
        """
        self.target_mode = target_mode
        self.min_samples_per_class = min_samples_per_class
    
    def balance(self,
                pure_samples: Dict[int, List[int]],
                multi_samples: Optional[List[Tuple[int, List[int]]]] = None,
                seed: int = 42) -> Dict:
        """
        Balance samples across 4 classes.
        
        Args:
            pure_samples: Dict mapping class label (0-3) to list of sample indices
                         {0: [idx1, idx2, ...], 1: [...], 2: [...], 3: [...]}
            multi_samples: Optional list of (idx, possible_classes) for ambiguous samples
                          [(idx, [1, 2]), (idx, [2, 3]), ...]
            seed: Random seed for reproducibility
        
        Returns:
            Dictionary containing:
                - 'balanced_indices': {0: [...], 1: [...], 2: [...], 3: [...]}
                - 'target_count': int (samples per class)
                - 'stats': detailed statistics
        """
        random.seed(seed)
        np.random.seed(seed)
        
        # Validate input
        for cls in [0, 1, 2, 3]:
            if cls not in pure_samples:
                raise ValueError(f"Missing class {cls} in pure_samples")
        
        # Step 1: Count pure samples per class
        pure_counts = {cls: len(indices) for cls, indices in pure_samples.items()}
        
        # Step 2: Determine target count
        if self.target_mode == 'auto':
            target_count = min(pure_counts.values())
            if target_count < self.min_samples_per_class:
                print(f"⚠️  Warning: Minimum class has only {target_count} samples")
        else:
            target_count = int(self.target_mode)
        
        # Step 3: Allocate pure samples first
        balanced = {}
        gaps = {}  # Track how many more samples each class needs
        
        for cls in [0, 1, 2, 3]:
            if pure_counts[cls] >= target_count:
                # Undersample: randomly select target_count samples
                balanced[cls] = random.sample(pure_samples[cls], target_count)
                gaps[cls] = 0
            else:
                # Use all pure samples + need more
                balanced[cls] = pure_samples[cls].copy()
                gaps[cls] = target_count - len(balanced[cls])
        
        # Step 4: Fill gaps with multi-label samples (if available)
        multi_used = 0
        if multi_samples and any(gaps.values()):
            # Sort classes by gap size (descending) - prioritize largest gaps
            gap_priority = sorted(gaps.items(), key=lambda x: x[1], reverse=True)
            
            used_indices = set()
            
            for cls, gap_size in gap_priority:
                if gap_size == 0:
                    continue
                
                # Find multi-label samples assignable to this class
                candidates = [
                    idx for idx, possible_classes in multi_samples
                    if cls in possible_classes and idx not in used_indices
                ]
                
                # Assign as many as needed (or available)
                to_assign = min(gap_size, len(candidates))
                if to_assign > 0:
                    selected = random.sample(candidates, to_assign)
                    balanced[cls].extend(selected)
                    used_indices.update(selected)
                    multi_used += to_assign
        
        # Step 5: Final statistics
        final_counts = {cls: len(indices) for cls, indices in balanced.items()}
        gaps_remaining = {cls: max(0, target_count - final_counts[cls]) for cls in [0, 1, 2, 3]}
        
        return {
            'balanced_indices': balanced,
            'target_count': target_count,
            'stats': {
                'pure_counts': pure_counts,
                'final_counts': final_counts,
                'multi_available': len(multi_samples) if multi_samples else 0,
                'multi_used': multi_used,
                'gaps_remaining': gaps_remaining,
                'fully_balanced': all(v == 0 for v in gaps_remaining.values())
            }
        }
    
    def get_class_distribution(self, balanced_indices: Dict[int, List[int]]) -> str:
        """
        Get human-readable distribution summary.
        
        Args:
            balanced_indices: Output from balance() method
        
        Returns:
            Formatted string with class distribution
        """
        class_names = {0: 'Negative', 1: 'Police', 2: 'Ambulance', 3: 'Fire'}
        counts = {cls: len(indices) for cls, indices in balanced_indices.items()}
        
        lines = []
        for cls in [0, 1, 2, 3]:
            lines.append(f"  Class {cls} ({class_names[cls]}): {counts[cls]} samples")
        lines.append(f"  Total: {sum(counts.values())} samples")
        
        return '\n'.join(lines)


def parse_audioset_multi_labels(df, mid_mapping: Dict[str, int], label_column: str = 'positive_labels'):
    """
    Parse AudioSet-style CSV with multi-label annotations.
    
    Args:
        df: Pandas DataFrame with AudioSet metadata
        mid_mapping: Dict mapping AudioSet MIDs to class labels
                    Can be:
                    - {'/m/04qvtq': 1, ...} (simple int)
                    - {'/m/04qvtq': [1, 'Police car (siren)'], ...} (array format)
        label_column: Name of column containing label lists (default: 'positive_labels')
    
    Returns:
        Dictionary with:
            - 'pure': {1: [idx, ...], 2: [...], 3: [...]}  (single-label samples)
            - 'multi': [(idx, [1, 2]), ...]  (multi-label samples)
            - 'stats': parsing statistics
    """
    import ast
    
    # Normalize mid_mapping to simple format {mid: label_int}
    normalized_mapping = {}
    for mid, value in mid_mapping.items():
        if isinstance(value, list):
            # Array format: [label, name]
            normalized_mapping[mid] = value[0]
        else:
            # Simple int format
            normalized_mapping[mid] = value
    
    pure = {1: [], 2: [], 3: []}
    multi = []
    skipped = 0
    
    for idx, row in df.iterrows():
        try:
            # Parse label string (e.g., "['/m/04qvtq', '/m/012n7d']")
            if isinstance(row[label_column], str):
                labels_list = ast.literal_eval(row[label_column])
            else:
                labels_list = row[label_column]
            
            # Map to class labels (1, 2, 3) using normalized mapping
            ev_classes = [normalized_mapping[mid] for mid in labels_list if mid in normalized_mapping]
            
            if len(ev_classes) == 1:
                # Pure sample - assign to single class
                pure[ev_classes[0]].append(idx)
            elif len(ev_classes) > 1:
                # Multi-label sample - keep for gap filling
                multi.append((idx, sorted(ev_classes)))
            else:
                # No EV labels found
                skipped += 1
        
        except Exception as e:
            skipped += 1
            continue
    
    stats = {
        'pure_counts': {cls: len(indices) for cls, indices in pure.items()},
        'multi_count': len(multi),
        'skipped': skipped,
        'total_processed': len(df)
    }
    
    return {'pure': pure, 'multi': multi, 'stats': stats}


def get_class_names_from_mapping(mid_mapping: Dict) -> Dict[int, str]:
    """
    Extract class names from AudioSet mapping.
    
    Args:
        mid_mapping: Dict with AudioSet MIDs
                    Can be:
                    - {'/m/04qvtq': [1, 'Police car (siren)'], ...} (array format)
                    - {'/m/04qvtq': 1, ...} (fallback to generic names)
    
    Returns:
        Dict mapping class labels to names: {1: 'Police car (siren)', ...}
    """
    class_names = {}
    
    for mid, value in mid_mapping.items():
        if isinstance(value, list):
            # Array format: [label, name]
            label, name = value
            class_names[label] = name
        else:
            # Fallback: use generic names
            generic_names = {0: 'Negative', 1: 'Police', 2: 'Ambulance', 3: 'Fire'}
            label = value
            class_names[label] = generic_names.get(label, f'Class {label}')
    
    return class_names


def print_balance_summary(result: Dict, title: str = "4-Way Balance Summary"):
    """
    Print formatted summary of balancing results.
    
    Args:
        result: Output from FourWayBalancer.balance()
        title: Title for the summary
    """
    print("\n" + "─" * 80)
    print(title.upper())
    print("─" * 80)
    
    stats = result['stats']
    class_names = {0: 'Negative', 1: 'Police', 2: 'Ambulance', 3: 'Fire'}
    
    print(f"\nTarget: {result['target_count']} samples per class")
    
    print("\nInitial (pure samples only):")
    for cls in [0, 1, 2, 3]:
        print(f"  {class_names[cls]:12s}: {stats['pure_counts'][cls]:4d} samples")
    
    if stats['multi_available'] > 0:
        print(f"\nMulti-label samples:")
        print(f"  Available: {stats['multi_available']}")
        print(f"  Used: {stats['multi_used']}")
    
    print("\nFinal distribution:")
    for cls in [0, 1, 2, 3]:
        count = stats['final_counts'][cls]
        gap = stats['gaps_remaining'][cls]
        status = "✓" if gap == 0 else f"⚠️ (-{gap})"
        print(f"  {class_names[cls]:12s}: {count:4d} samples {status}")
    
    print(f"\nTotal samples: {sum(stats['final_counts'].values())}")
    print(f"Fully balanced: {'Yes ✓' if stats['fully_balanced'] else 'No (see gaps above)'}")
    print("─" * 80)
