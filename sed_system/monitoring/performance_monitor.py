"""Performance Monitor for SED System
======================================
Monitors CPU, RAM, and throughput during inference.

Author: Stefano Giacomelli - Ph.D. candidate in ICT (DISIM dpt. - University of L'Aquila)
"""

import os
import time
import logging
import threading
from typing import Dict, Optional

import psutil


class PerformanceMonitor:
    """
    Monitor system performance during inference.
    
    Tracks:
    - CPU usage (%)
    - RAM usage (MB)
    - Throughput (audio duration / processing time)
    """
    
    def __init__(self, sampling_interval: float = 0.1):
        """
        Initialize performance monitor.
        
        Args:
            sampling_interval: How often to sample CPU/RAM (seconds)
        """
        self.sampling_interval = sampling_interval
        
        # Process handle
        self.process = psutil.Process(os.getpid())
        
        # Monitoring state
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        
        # Data storage
        self.cpu_samples = []
        self.ram_samples = []
        
        # Timing
        self.start_time = None
        self.end_time = None
        
        # Audio duration (set externally)
        self.audio_duration = None
    
    def start(self):
        """Start monitoring in background thread."""
        if self.monitoring:
            logging.warning("Performance monitor already running")
            return
        
        self.monitoring = True
        self.start_time = time.perf_counter()
        self.cpu_samples = []
        self.ram_samples = []
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        logging.info("Performance monitoring started")
    
    def stop(self):
        """Stop monitoring."""
        if not self.monitoring:
            logging.warning("Performance monitor not running")
            return
        
        self.monitoring = False
        self.end_time = time.perf_counter()
        
        # Wait for thread to finish
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        
        logging.info("Performance monitoring stopped")
    
    def _monitor_loop(self):
        """Background loop to sample CPU and RAM."""
        while self.monitoring:
            try:
                # Sample CPU (percent)
                cpu_percent = self.process.cpu_percent(interval=None)
                self.cpu_samples.append(cpu_percent)
                
                # Sample RAM (MB)
                ram_mb = self.process.memory_info().rss / (1024 * 1024)
                self.ram_samples.append(ram_mb)
                
            except Exception as e:
                logging.error(f"Error sampling performance: {e}")
            
            time.sleep(self.sampling_interval)
    
    def set_audio_duration(self, duration: float):
        """
        Set total audio duration for throughput calculation.
        
        Args:
            duration: Audio duration in seconds
        """
        self.audio_duration = duration
    
    def get_stats(self) -> Dict:
        """
        Get performance statistics.
        
        Returns:
            Dictionary with CPU, RAM, throughput stats
        """
        if self.start_time is None:
            return {}
        
        # Total time
        if self.end_time is None:
            total_time = time.perf_counter() - self.start_time
        else:
            total_time = self.end_time - self.start_time
        
        stats = {
            'total_time': total_time,
            'total_samples': len(self.cpu_samples)
        }
        
        # CPU stats
        if self.cpu_samples:
            stats['cpu'] = {
                'min': min(self.cpu_samples),
                'max': max(self.cpu_samples),
                'mean': sum(self.cpu_samples) / len(self.cpu_samples)
            }
        
        # RAM stats
        if self.ram_samples:
            stats['ram_mb'] = {
                'min': min(self.ram_samples),
                'max': max(self.ram_samples),
                'mean': sum(self.ram_samples) / len(self.ram_samples)
            }
        
        # Throughput (real-time factor)
        if self.audio_duration is not None and total_time > 0:
            throughput = self.audio_duration / total_time
            stats['throughput'] = throughput
            stats['audio_duration'] = self.audio_duration
            stats['is_realtime_capable'] = throughput >= 1.0
        
        return stats
    
    def get_summary(self) -> str:
        """
        Get human-readable summary.
        
        Returns:
            Formatted summary string
        """
        stats = self.get_stats()
        
        if not stats:
            return "No performance data available"
        
        lines = [
            f"Performance Summary:",
            f"  Total time: {stats['total_time']:.2f}s"
        ]
        
        if 'cpu' in stats:
            lines.append(f"  CPU: {stats['cpu']['mean']:.1f}% (max: {stats['cpu']['max']:.1f}%)")
        
        if 'ram_mb' in stats:
            lines.append(f"  RAM: {stats['ram_mb']['mean']:.1f} MB (max: {stats['ram_mb']['max']:.1f} MB)")
        
        if 'throughput' in stats:
            rt = "✓" if stats['is_realtime_capable'] else "✗"
            lines.append(f"  Throughput: {stats['throughput']:.2f}x real-time {rt}")
        
        return "\n".join(lines)
