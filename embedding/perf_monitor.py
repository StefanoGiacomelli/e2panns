import psutil
import time
import logging
import threading
from globals import audio_file_path

import psutil
import time
import logging
import threading

class PerformanceMonitor:
    def __init__(self, start_time, interval=1.0, log_file="./performance_metrics.log"):
        """
        Initialize the performance monitor.
        Args:
            start_time (float): The starting time of the process (from time.perf_counter()).
            interval (float): Time interval in seconds between performance checks.
            log_file (str): Path to save the performance log.
        """
        self.interval = interval
        self.log_file = log_file
        self.running = False
        self.thread = None
        self.start_time = start_time

    def start(self):
        """Start monitoring in a separate thread."""
        if self.running:
            logging.warning("Performance monitor is already running.")
            return
        self.running = True
        self.thread = threading.Thread(target=self._monitor)
        self.thread.start()
        logging.info("Performance monitor started.")

    def stop(self):
        """Stop the monitoring thread."""
        if not self.running:
            logging.warning("Performance monitor is not running.")
            return
        self.running = False
        if self.thread is not None:
            self.thread.join()
            logging.info("Performance monitor stopped.")

    def _monitor(self):
        """Monitor system performance and log metrics."""
        with open(self.log_file, "w") as log:
            # Write the header with dynamic core count
            num_cores = psutil.cpu_count(logical=True)
            core_headers = ",".join([f"Core{i}(%)" for i in range(num_cores)])
            log.write(f"Time,Total_CPU(%),{core_headers},Memory(%),Available Memory(MB)\n")

            while self.running:
                # Get relative time
                current_time = time.perf_counter()
                relative_time = current_time - self.start_time

                # Get CPU load
                total_cpu_load = psutil.cpu_percent()
                per_core_loads = psutil.cpu_percent(percpu=True)  # List of core loads

                # Get memory info
                memory_info = psutil.virtual_memory()
                memory_load = memory_info.percent
                available_memory = memory_info.available / (1024 * 1024)  # Convert to MB

                # Log the metrics
                per_core_str = ",".join(f"{load:.2f}" for load in per_core_loads)
                log_line = f"{relative_time:.2f},{total_cpu_load:.2f},{per_core_str},{memory_load:.2f},{available_memory:.2f}\n"
                log.write(log_line)
                log.flush()

                # Debug print (optional)
                logging.debug(f"Time: {relative_time:.2f}s, Total CPU: {total_cpu_load}%, Per-Core: {per_core_loads}, "
                              f"Memory: {memory_load}%, Available: {available_memory:.2f}MB")

                time.sleep(self.interval)
