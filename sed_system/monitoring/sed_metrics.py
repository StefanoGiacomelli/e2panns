"""SED Metrics for Sound Event Detection
=========================================
Computes standard SED metrics using sed_eval library.

Author: Stefano Giacomelli - Ph.D. candidate in ICT (DISIM dpt. - University of L'Aquila)
"""

import logging
from typing import List, Dict, Tuple, Optional
from pathlib import Path

import numpy as np


class SEDMetrics:
    """
    Sound Event Detection metrics calculator.
    
    Features:
    - Convert probability predictions to events (onset/offset)
    - Save events in DCASE format
    - Compute Precision/Recall/F1 with sed_eval (if ground truth available)
    """
    
    def __init__(self, class_name: str = "Emergency_vehicle", 
                 threshold: float = 0.5,
                 min_duration: float = 0.1,
                 merge_gap: float = 0.2):
        """
        Initialize SED metrics calculator.
        
        Args:
            class_name: Event class name
            threshold: Probability threshold for detection
            min_duration: Minimum event duration (seconds)
            merge_gap: Merge events closer than this gap (seconds)
        """
        self.class_name = class_name
        self.threshold = threshold
        self.min_duration = min_duration
        self.merge_gap = merge_gap
        
        self.predictions: List[Dict] = []
    
    def add_prediction(self, timestamp: float, probability: float):
        """
        Add a prediction at given timestamp.
        
        Args:
            timestamp: Audio timestamp (seconds)
            probability: Model prediction (0-1)
        """
        self.predictions.append({
            'timestamp': timestamp,
            'probability': probability
        })
    
    def add_predictions_batch(self, predictions: List[Dict]):
        """
        Add multiple predictions at once.
        
        Args:
            predictions: List of dicts with 'timestamp' and 'probability'
        """
        self.predictions.extend(predictions)
    
    def get_detected_events(self) -> List[Tuple[float, float, float]]:
        """
        Extract events from predictions using threshold.
        
        Returns:
            List of (onset, offset, confidence) tuples
        """
        if not self.predictions:
            return []
        
        # Sort by timestamp
        sorted_preds = sorted(self.predictions, key=lambda x: x['timestamp'])
        
        events = []
        in_event = False
        onset = None
        max_conf = 0.0
        
        for pred in sorted_preds:
            ts = pred['timestamp']
            prob = pred['probability']
            
            if prob >= self.threshold:
                if not in_event:
                    # Event start
                    onset = ts
                    max_conf = prob
                    in_event = True
                else:
                    # Event continues, update max confidence
                    max_conf = max(max_conf, prob)
            else:
                if in_event:
                    # Event end
                    offset = ts
                    duration = offset - onset
                    
                    if duration >= self.min_duration:
                        events.append((onset, offset, max_conf))
                    
                    in_event = False
        
        # Handle case where event extends to end
        if in_event and onset is not None:
            offset = sorted_preds[-1]['timestamp']
            duration = offset - onset
            if duration >= self.min_duration:
                events.append((onset, offset, max_conf))
        
        # Merge close events
        events = self._merge_close_events(events)
        
        return events
    
    def _merge_close_events(self, events: List[Tuple[float, float, float]]) -> List[Tuple[float, float, float]]:
        """
        Merge events that are closer than merge_gap.
        
        Args:
            events: List of (onset, offset, confidence) tuples
        
        Returns:
            Merged events
        """
        if len(events) <= 1:
            return events
        
        merged = []
        current_onset, current_offset, current_conf = events[0]
        
        for onset, offset, conf in events[1:]:
            gap = onset - current_offset
            
            if gap <= self.merge_gap:
                # Merge with current event
                current_offset = offset
                current_conf = max(current_conf, conf)
            else:
                # Save current and start new
                merged.append((current_onset, current_offset, current_conf))
                current_onset, current_offset, current_conf = onset, offset, conf
        
        # Add last event
        merged.append((current_onset, current_offset, current_conf))
        
        return merged
    
    def save_events(self, output_path: str):
        """
        Save detected events in DCASE format.
        
        Format: onset \t offset \t event_label
        
        Args:
            output_path: Path to output file
        """
        events = self.get_detected_events()
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            for onset, offset, conf in events:
                f.write(f"{onset:.3f}\t{offset:.3f}\t{self.class_name}\n")
        
        logging.info(f"Saved {len(events)} events to {output_path}")
        return str(output_path)
    
    def compute_metrics(self, ground_truth_events: List[Tuple[float, float]],
                       time_resolution: float = 0.01,
                       tolerance: float = 0.2) -> Dict:
        """
        Compute SED metrics against ground truth.
        
        Args:
            ground_truth_events: List of (onset, offset) tuples
            time_resolution: Time resolution for segment-based metrics (seconds)
            tolerance: Tolerance for event-based metrics (seconds)
        
        Returns:
            Dictionary with precision, recall, F1
        """
        try:
            import sed_eval
        except ImportError as e:
            logging.error(f"sed_eval not installed: {e}. Install with: pip install sed_eval")
            return self._compute_simple_metrics(ground_truth_events)
        
        try:
            # Convert to sed_eval format (list of dicts)
            detected_events = self.get_detected_events()
            
            reference = []
            for onset, offset in ground_truth_events:
                reference.append({
                    'event_onset': onset,
                    'event_offset': offset,
                    'event_label': self.class_name,
                    'file': 'audio.wav'
                })
            
            estimated = []
            for onset, offset, conf in detected_events:
                estimated.append({
                    'event_onset': onset,
                    'event_offset': offset,
                    'event_label': self.class_name,
                    'file': 'audio.wav'
                })
            
            # Segment-based metrics
            segment_evaluator = sed_eval.sound_event.SegmentBasedMetrics(
                event_label_list=[self.class_name],
                time_resolution=time_resolution
            )
            segment_evaluator.evaluate(
                reference_event_list=reference,
                estimated_event_list=estimated
            )
            
            # Event-based metrics
            event_evaluator = sed_eval.sound_event.EventBasedMetrics(
                event_label_list=[self.class_name],
                t_collar=tolerance
            )
            event_evaluator.evaluate(
                reference_event_list=reference,
                estimated_event_list=estimated
            )
            
            # Extract results
            segment_results = segment_evaluator.results_class_wise_metrics()[self.class_name]
            event_results = event_evaluator.results_class_wise_metrics()[self.class_name]
            
            metrics = {
                'segment_based': {
                    # F-measure metrics
                    'precision': segment_results['f_measure']['precision'],
                    'recall': segment_results['f_measure']['recall'],
                    'f1': segment_results['f_measure']['f_measure'],
                    # Accuracy metrics
                    'accuracy': segment_results['accuracy']['accuracy'],
                    'balanced_accuracy': segment_results['accuracy']['balanced_accuracy'],
                    'sensitivity': segment_results['accuracy']['sensitivity'],
                    'specificity': segment_results['accuracy']['specificity'],
                    # Error rate metrics
                    'error_rate': segment_results['error_rate']['error_rate'],
                    'deletion_rate': segment_results['error_rate']['deletion_rate'],
                    'insertion_rate': segment_results['error_rate']['insertion_rate'],
                    'substitution_rate': segment_results['error_rate'].get('substitution_rate', 0.0)
                },
                'event_based': {
                    # F-measure metrics
                    'precision': event_results['f_measure']['precision'],
                    'recall': event_results['f_measure']['recall'],
                    'f1': event_results['f_measure']['f_measure'],
                    # Error rate metrics (event-based doesn't have accuracy)
                    'error_rate': event_results.get('error_rate', {}).get('error_rate', None),
                    'deletion_rate': event_results.get('error_rate', {}).get('deletion_rate', None),
                    'insertion_rate': event_results.get('error_rate', {}).get('insertion_rate', None),
                    'substitution_rate': event_results.get('error_rate', {}).get('substitution_rate', None)
                },
                'num_detected': len(detected_events),
                'num_ground_truth': len(ground_truth_events)
            }
            
            return metrics
            
        except Exception as e:
            logging.error(f"Error computing sed_eval metrics: {e}", exc_info=True)
            logging.info("Falling back to simple metrics")
            return self._compute_simple_metrics(ground_truth_events)
    
    def _compute_simple_metrics(self, ground_truth_events: List[Tuple[float, float]]) -> Dict:
        """
        Compute simplified metrics without sed_eval (fallback).
        
        Args:
            ground_truth_events: List of (onset, offset) tuples
        
        Returns:
            Dictionary with basic metrics
        """
        detected_events = self.get_detected_events()
        
        # Simple overlap-based matching
        tp = 0
        for gt_onset, gt_offset in ground_truth_events:
            for det_onset, det_offset, _ in detected_events:
                # Check for overlap
                overlap = min(gt_offset, det_offset) - max(gt_onset, det_onset)
                if overlap > 0:
                    tp += 1
                    break
        
        fp = len(detected_events) - tp
        fn = len(ground_truth_events) - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'simple': {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'tp': tp,
                'fp': fp,
                'fn': fn
            },
            'num_detected': len(detected_events),
            'num_ground_truth': len(ground_truth_events)
        }
