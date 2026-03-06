"""Metrics Logger for SED System
=================================
Logs inference results to CSV and JSON for analysis and reproducibility.

Author: Stefano Giacomelli - Ph.D. candidate in ICT (DISIM dpt. - University of L'Aquila)
"""

import os
import csv
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional


class MetricsLogger:
    """
    Logger for inference metrics with CSV and JSON export.
    
    Records:
    - Timestamp of each inference
    - Probability prediction
    - Inference duration
    - Frame size
    
    Exports:
    - CSV: All inference results for analysis
    - JSON: Metadata + aggregated statistics
    """
    
    def __init__(self, output_dir: Optional[str] = None, experiment_name: str = "experiment"):
        """
        Initialize metrics logger.
        
        Args:
            output_dir: Directory to save results. If None, creates timestamped directory.
            experiment_name: Name of experiment for directory naming
        """
        # Create output directory
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"results/{experiment_name}_{timestamp}"
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Storage for results
        self.results: List[Dict] = []
        
        # Metadata
        self.metadata = {
            'experiment_name': experiment_name,
            'start_time': datetime.now().isoformat(),
            'output_dir': str(self.output_dir)
        }
        
        logging.info(f"MetricsLogger initialized: {self.output_dir}")
    
    def log_inference(self, timestamp: float, probability: float, 
                     duration: float, frame_size: int):
        """
        Log a single inference result.
        
        Args:
            timestamp: Audio timestamp (seconds)
            probability: Model prediction (0-1)
            duration: Inference time (seconds)
            frame_size: Number of audio samples in frame
        """
        result = {
            'timestamp': timestamp,
            'probability': probability,
            'duration': duration,
            'frame_size': frame_size
        }
        self.results.append(result)
    
    def set_metadata(self, **kwargs):
        """
        Add metadata fields.
        
        Args:
            **kwargs: Key-value pairs to add to metadata
        """
        self.metadata.update(kwargs)
    
    def get_statistics(self) -> Dict:
        """
        Compute aggregated statistics from logged results.
        
        Returns:
            Dictionary with statistics
        """
        if not self.results:
            return {}
        
        probabilities = [r['probability'] for r in self.results]
        durations = [r['duration'] for r in self.results]
        frame_sizes = [r['frame_size'] for r in self.results]
        
        stats = {
            'total_inferences': len(self.results),
            'probability': {
                'min': min(probabilities),
                'max': max(probabilities),
                'mean': sum(probabilities) / len(probabilities),
            },
            'inference_duration': {
                'min': min(durations),
                'max': max(durations),
                'mean': sum(durations) / len(durations),
                'total': sum(durations)
            },
            'frame_size': {
                'min': min(frame_sizes),
                'max': max(frame_sizes),
                'mean': sum(frame_sizes) / len(frame_sizes)
            }
        }
        
        # Count high confidence predictions
        high_conf_threshold = 0.5
        high_conf_count = sum(1 for p in probabilities if p >= high_conf_threshold)
        stats['high_confidence_count'] = high_conf_count
        stats['high_confidence_ratio'] = high_conf_count / len(probabilities)
        
        return stats
    
    def save_results(self) -> Dict[str, str]:
        """
        Save results to CSV and JSON files.
        
        Returns:
            Dictionary with paths to saved files
        """
        if not self.results:
            logging.warning("No results to save")
            return {}
        
        # Save CSV
        csv_path = self.output_dir / "inference_results.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['timestamp', 'probability', 'duration', 'frame_size'])
            writer.writeheader()
            writer.writerows(self.results)
        logging.info(f"Saved CSV: {csv_path}")
        
        # Save JSON with metadata and statistics
        self.metadata['end_time'] = datetime.now().isoformat()
        self.metadata['statistics'] = self.get_statistics()
        
        json_path = self.output_dir / "metadata.json"
        with open(json_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        logging.info(f"Saved JSON: {json_path}")
        
        return {
            'csv': str(csv_path),
            'json': str(json_path)
        }
    
    def load_results(self, csv_path: str):
        """
        Load results from CSV file.
        
        Args:
            csv_path: Path to CSV file
        """
        self.results = []
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.results.append({
                    'timestamp': float(row['timestamp']),
                    'probability': float(row['probability']),
                    'duration': float(row['duration']),
                    'frame_size': int(row['frame_size'])
                })
        logging.info(f"Loaded {len(self.results)} results from {csv_path}")
