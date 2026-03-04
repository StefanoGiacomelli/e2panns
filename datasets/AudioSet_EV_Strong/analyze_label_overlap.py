"""
Analyze EV Label Overlap in AudioSet Strong Metadata
=====================================================
This script analyzes:
- Distribution of single vs multiple EV labels per segment
- Overlap between root label (Emergency vehicle) and leaf labels
- All possible label combinations

Author: Stefano Giacomelli - Ph.D. candidate in ICT (DISIM dpt. - University of L'Aquila)
"""

from collections import defaultdict, Counter

# MID to human-readable mapping
MID_TO_NAME = {
    '/m/03j1ly': 'Emergency vehicle',
    '/m/04qvtq': 'Police car (siren)',
    '/m/012n7d': 'Ambulance (siren)',
    '/m/012ndj': 'Fire engine, fire truck (siren)'
}

EV_MIDS = set(MID_TO_NAME.keys())

train_path = 'audioset_strong_metadata/audioset_train_strong.tsv'
eval_path = 'audioset_strong_metadata/audioset_eval_strong.tsv'

# Parse metadata
segment_labels = defaultdict(set)

for path in [train_path, eval_path]:
    with open(path, 'r') as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 4:
                segment_id, start, end, mid = parts
                if mid in EV_MIDS:
                    segment_labels[segment_id].add(mid)


print('='*80)
print('ANALISI MULTIPLE LABEL EV PER SEGMENT')
print('='*80)

# Count segments by number of unique EV labels
label_counts = Counter()
for segment, labels in segment_labels.items():
    label_counts[len(labels)] += 1

print(f'\nSegmenti con eventi EV: {len(segment_labels)}')
print(f'\nDistribuzione per numero di label EV uniche:')
for num_labels, count in sorted(label_counts.items()):
    print(f'  {num_labels} label EV: {count} segmenti ({count/len(segment_labels)*100:.1f}%)')

# ============================================================================
# OVERLAP ROOT vs LEAF
# ============================================================================
print('\n' + '='*80)
print('OVERLAP ROOT (Emergency vehicle) vs LEAF LABELS')
print('='*80)

root_mid = '/m/03j1ly'
leaf_mids = {'/m/04qvtq', '/m/012n7d', '/m/012ndj'}

segments_with_root = {seg for seg, labels in segment_labels.items() if root_mid in labels}
segments_with_leaf = {seg for seg, labels in segment_labels.items() if labels & leaf_mids}
segments_with_both = segments_with_root & segments_with_leaf

print(f'\nSegmenti con root "Emergency vehicle": {len(segments_with_root)}')
print(f'Segmenti con label leaf (Police/Ambulance/Fire): {len(segments_with_leaf)}')
print(f'Segmenti con ENTRAMBE (root + leaf): {len(segments_with_both)}')

# ============================================================================
# LISTA COMPLETA CAMPIONI CON MULTIPLE LABEL
# ============================================================================
print('\n' + '='*80)
print('LISTA COMPLETA CAMPIONI CON MULTIPLE LABEL EV')
print('='*80)

multi_label_segments = [(seg, labels) for seg, labels in segment_labels.items() if len(labels) > 1]
multi_label_segments.sort(key=lambda x: (len(x[1]), x[0]), reverse=True)

if multi_label_segments:
    print(f'\nTotale segmenti con multiple label: {len(multi_label_segments)}')
    print(f'\nLista completa (ordinata per numero di label):')
    print('-' * 80)
    
    for seg, mids in multi_label_segments:
        # Convert MIDs to human-readable names
        label_names = sorted([MID_TO_NAME[mid] for mid in mids])
        print(f'  {seg}')
        print(f'    → {len(mids)} label: {", ".join(label_names)}')
else:
    print('\n  Nessun segmento ha multiple label EV!')

# ============================================================================
# TUTTE LE COMBINAZIONI DI LABEL
# ============================================================================
print('\n' + '='*80)
print('CONTEGGIO DI TUTTE LE COMBINAZIONI DI LABEL')
print('='*80)

# Create frozensets for counting combinations
label_combinations = Counter()
for segment, mids in segment_labels.items():
    # Create a sorted tuple of label names for the combination
    label_names = tuple(sorted([MID_TO_NAME[mid] for mid in mids]))
    label_combinations[label_names] += 1

print(f'\nTotale combinazioni uniche: {len(label_combinations)}')
print(f'\nDistribuzione completa:\n')

# Sort by frequency (descending) then by combination size
sorted_combinations = sorted(label_combinations.items(), 
                            key=lambda x: (-x[1], len(x[0])))

for labels_tuple, count in sorted_combinations:
    percentage = count / len(segment_labels) * 100
    
    # Format label combination
    if len(labels_tuple) == 1:
        combo_str = f"Solo: {labels_tuple[0]}"
    else:
        combo_str = f"Combo ({len(labels_tuple)}): {' + '.join(labels_tuple)}"
    
    print(f'  {combo_str}')
    print(f'    → {count} segmenti ({percentage:.1f}%)')
    print()

# ============================================================================
# CONTEGGIO EVENTI (non segmenti)
# ============================================================================
print('='*80)
print('CONTEGGIO EVENTI TOTALI PER TIPO')
print('='*80)

event_counts = Counter()
for path in [train_path, eval_path]:
    with open(path, 'r') as f:
        next(f)
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 4:
                segment_id, start, end, mid = parts
                if mid in EV_MIDS:
                    event_counts[mid] += 1

print(f'\nTotale eventi EV per tipo:')
total_events = 0
for mid in sorted(event_counts.keys(), key=lambda x: event_counts[x], reverse=True):
    count = event_counts[mid]
    label_name = MID_TO_NAME[mid]
    percentage = count / sum(event_counts.values()) * 100
    print(f'  {label_name}: {count} eventi ({percentage:.1f}%)')
    total_events += count

print(f'\n  TOTALE EVENTI EV: {total_events}')
print('='*80)
