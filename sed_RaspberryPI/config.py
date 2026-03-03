import random
import numpy as np
import torch
import logging

import matplotlib.pyplot as plt
from pathlib import Path

from utils import HousekeepingLogger
from globals import HKL_PATH, RANDOM_SEED, AUDIO_FILE_PATH, ENABLE_MONITORING, MONITORING_INTERVAL, HK_LOGGING_ENABLED, TORCH_NUM_THREADS
from perf_monitor import PerformanceMonitor

def init_env(audio_file_path, checkpoint_path):
    """
    Initializes random generators.
    Configures logging and loglevel.
    Instanciates HousekeepingLogger object and sets the start timestamp for housekeeping functions.
    Instanciates and starts performance monitor.

    Returns:
        HousekeepingLogger: the housekeeping object
        PerformanceMonitor: the perf monitor object
    """

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    
    plt.rcParams['mathtext.fontset'] = 'cm'
    plt.rcParams['font.family'] = 'STIXGeneral'
    plt.rc('font', size=12)

    torch.set_num_threads(TORCH_NUM_THREADS)

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    #logging.getLogger().setLevel(logging.INFO) ############### SET LOGGING LEVEL HERE
    logging.disable(logging.CRITICAL)

    hk_logger = HousekeepingLogger(
        checkpoint_path = checkpoint_path,
        audio_path = audio_file_path,
        output_base_dir = HKL_PATH,        
        logging_enabled=HK_LOGGING_ENABLED) # also starts the overall timing

    hk_logger.start() # init reference timestamp

    # Initialize and start performance monitoring
    monitor = None
    if ENABLE_MONITORING:
        audio_stem = Path(audio_file_path).stem
        checkpoint_name = Path(checkpoint_path).name
        perf_dir = Path(HKL_PATH) / checkpoint_name / "perf"
        perf_dir.mkdir(parents=True, exist_ok=True)
        perf_file = perf_dir / f"{audio_stem}.csv"

        monitor = PerformanceMonitor(start_time=hk_logger.t_start, interval=MONITORING_INTERVAL, log_file=perf_file)
        monitor.start()

    return hk_logger, monitor