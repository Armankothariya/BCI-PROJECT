# =================================================================
# modules/data_acquisition.py
#
# BCI DATA ACQUISITION AND SIMULATION MODULE
# This module provides a unified interface for acquiring EEG data,
# whether from live hardware (via BrainFlow) or by simulating a
# real-time stream from pre-recorded datasets (DEAP, SEED, etc.).
#
# Key features:
# - Hardware-agnostic design (supports any BrainFlow-compatible board).
# - Realistic dataset-based streaming for reproducible demos.
# - Clean, generator-based interface for real-time applications.
# =================================================================

import numpy as np
import time
import h5py
import pandas as pd
from datetime import datetime
import logging
import glob
import os
import random

# --- Try to import BrainFlow for live hardware support ---
# If BrainFlow isn't installed, the project can still run in dataset simulation mode.
try:
    from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
    from brainflow.data_filter import DataFilter
    _brainflow_ok = True
except ImportError:
    _brainflow_ok = False
    logging.warning("BrainFlow not found. Live hardware mode is disabled. "
                    "Only dataset simulation will be available.")

# --- Define Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EEGStreamer:
    """
    A class to handle EEG data streaming from either live hardware or datasets.
    It acts as a data generator, yielding windows of data for real-time processing.
    """
    def __init__(self, data_source="Simulated DEAP Data", window_size_s=2, sample_rate=128):
        """
        Initializes the data streamer.
        
        Args:
            data_source (str): The source of the EEG data. Can be "Simulated DEAP Data",
                               "Simulated Kaggle Data", "Synthetic BrainFlow Data", or
                               "Live Hardware (Experimental)".
            window_size_s (int): The duration of each data window in seconds.
            sample_rate (int): The sampling rate of the data in Hz.
        """
        self.data_source = data_source
        self.sample_rate = sample_rate
        self.window_size_s = window_size_s
        self.window_size = int(self.sample_rate * self.window_size_s)
        self.is_running = False
        
        # Determine mode and set up
        if "Simulated DEAP Data" in self.data_source:
            self.mode = "dataset_deap"
            self.dataset_path = "datasets/deap/data_preprocessed_python"
            self.data, self.labels = self._load_deap_dataset()
            self.current_trial_idx = 0
            self.current_frame_idx = 0
        elif "Simulated Kaggle Data" in self.data_source:
            self.mode = "dataset_kaggle"
            self.dataset_path = "datasets/kaggle_temp/emotion.csv"
            self.data, self.labels = self._load_kaggle_dataset()
            self.current_trial_idx = 0
            self.current_frame_idx = 0
        elif "Live Hardware" in self.data_source and _brainflow_ok:
            self.mode = "hardware"
            self.board_id = BoardIds.OPENBCI_CYTON_BOARD.value # Or any other supported board
            self.channels = BoardShim.get_eeg_channels(self.board_id)
            self._setup_brainflow()
        elif "Synthetic BrainFlow" in self.data_source and _brainflow_ok:
            self.mode = "synthetic"
            self.board_id = BoardIds.SYNTHETIC_BOARD.value
            self.channels = BoardShim.get_eeg_channels(self.board_id)
            self._setup_brainflow()
        else:
            self.mode = "offline_fallback"
            logger.error("Invalid data source or BrainFlow not available. Using a simple mock data generator.")
            self.channels = 32

    def _setup_brainflow(self):
        """Configures BrainFlow for real-time data streaming."""
        params = BrainFlowInputParams()
        # You can set parameters here for different boards (e.g., `params.serial_port`)
        self.board = BoardShim(self.board_id, params)
        self.eeg_channels = self.channels
        self.sample_rate = self.board.get_sampling_rate(self.board_id)
        self.window_size = int(self.sample_rate * self.window_size_s)
        self.board.prepare_session()
        
    def _load_deap_dataset(self):
        """Loads all DEAP dataset files from the specified directory."""
        all_trials = []
        all_labels = []
        
        file_list = sorted(glob.glob(os.path.join(self.dataset_path, "*.dat")))
        if not file_list:
            logger.error(f"No DEAP .dat files found in {self.dataset_path}.")
            return [], []
            
        # For simplicity in the demo, load one subject
        subject_path = file_list[0]
        logger.info(f"Loading data from DEAP subject: {os.path.basename(subject_path)}")
        
        data = h5py.File(subject_path, 'r')
        # Each trial is a 40-channel x 8064-sample array
        eeg_data = np.array(data['data'])[:, :32, :] # Use only the first 32 EEG channels
        # Get labels (valence, arousal)
        labels = np.array(data['labels'])
        
        all_trials = eeg_data
        all_labels = labels
        
        logger.info(f"DEAP dataset loaded. Shape: {all_trials.shape}. Simulating trials.")
        return all_trials, all_labels
    
    def _load_kaggle_dataset(self):
        """Loads the Kaggle dataset from a CSV file."""
        if not os.path.exists(self.dataset_path):
            logger.error(f"Kaggle dataset not found at {self.dataset_path}.")
            return [], []
            
        df = pd.read_csv(self.dataset_path)
        # Assuming the CSV has columns for EEG channels and a 'label' column
        eeg_cols = [col for col in df.columns if col.startswith('EEG')]
        eeg_data = df[eeg_cols].values
        labels = df['label'].values
        
        logger.info(f"Kaggle dataset loaded. Shape: {eeg_data.shape}. Simulating trial.")
        return [eeg_data.T], [labels] # Wrap in a list to match DEAP's trial structure

    def stream_data(self):
        """
        A generator that yields a new window of data on each call.
        This is the core of the real-time simulation.
        """
        self.is_running = True
        
        if self.mode in ["hardware", "synthetic"]:
            self.board.start_stream()
            logger.info(f"BrainFlow stream started. Mode: {self.mode}")
            while self.is_running:
                # Wait for a full window of data
                while self.board.get_board_data_count() < self.window_size:
                    time.sleep(0.005) 
                
                # Read the data and yield
                data = self.board.get_board_data()
                eeg_data = data[self.eeg_channels, -self.window_size:]
                
                yield eeg_data, datetime.now().isoformat()
        
        elif self.mode in ["dataset_deap", "dataset_kaggle"]:
            logger.info(f"Starting dataset simulation from {self.dataset_path}")
            while self.is_running:
                if self.current_trial_idx >= len(self.data):
                    logger.info("End of dataset reached. Looping back to the start.")
                    self.current_trial_idx = 0
                    self.current_frame_idx = 0

                current_trial = self.data[self.current_trial_idx]
                
                # Check for DEAP-specific shape (channels, samples)
                if self.mode == "dataset_deap":
                    current_trial = current_trial.T # Transpose to (samples, channels)
                
                if self.current_frame_idx + self.window_size > current_trial.shape[0]:
                    # Move to the next trial if the current one is finished
                    self.current_trial_idx += 1
                    self.current_frame_idx = 0
                    time.sleep(1) # Simulate a break between trials
                    continue
                
                window = current_trial[self.current_frame_idx : self.current_frame_idx + self.window_size, :]
                
                # Add a check for correct shape (samples, channels)
                if window.shape[0] != self.window_size:
                    # Pad the window if it's the last incomplete one
                    pad_size = self.window_size - window.shape[0]
                    window = np.pad(window, ((0, pad_size), (0, 0)), mode='constant')

                self.current_frame_idx += self.window_size
                
                yield window, datetime.now().isoformat()
                time.sleep(self.window_size_s * 0.5) # Simulate a half-speed stream for a smoother demo
        
        else: # Offline fallback mode
            while self.is_running:
                # Generate random data if BrainFlow and datasets are unavailable
                mock_data = np.random.randn(self.window_size, self.channels) * 10 
                yield mock_data, datetime.now().isoformat()
                time.sleep(1)

    def stop(self):
        """Stops the streaming process."""
        self.is_running = False
        if self.mode in ["hardware", "synthetic"] and _brainflow_ok:
            try:
                self.board.stop_stream()
                self.board.release_session()
                logger.info("BrainFlow session released.")
            except Exception as e:
                logger.error(f"Failed to stop BrainFlow session: {e}")
        logger.info("Data streamer stopped.")