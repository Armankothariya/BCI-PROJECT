# ============================================
# UNIVERSAL DATASET LOADER FOR BCI PROJECT
# (Research-Grade Version)
# - REMOVED: Scaling from the loader (now handled in run_pipeline.py after train-test split)
# - UPDATED: To work with new config structure
# - IMPROVED: Better error handling and logging
# ============================================

import os
import numpy as np
import pandas as pd
import scipy.io
import yaml
import joblib
from sklearn.preprocessing import LabelEncoder

from modules.feature_extraction import extract_features


class DatasetLoader:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)

        self.dataset_name = self.config["use_dataset"]
        self.dataset_config = self.config["datasets"][self.dataset_name]

        # Global options from new config structure
        global_config = self.config.get("global", {})
        self.apply_feature_extraction = global_config.get("apply_feature_extraction", False)
        self.include_bandpower = global_config.get("include_bandpower", True)
        self.include_stft = global_config.get("include_stft", False)
        self.include_entropy = global_config.get("include_entropy", False)  # <-- NEW
        self.include_wavelet = global_config.get("include_wavelet", False)  # <-- NEW
        self.per_channel_norm = global_config.get("per_channel_normalization", True)  # <-- NEW

        self.cross_dataset_enabled = self.config.get("cross_dataset", {}).get("enabled", False)

        # Use cache path from config, default to "cache"
        paths_config = self.config.get("paths", {})
        self.cache_dir = paths_config.get("cache_dir", "cache")
        os.makedirs(self.cache_dir, exist_ok=True)

    # ===========================
    # MAIN ENTRY
    # ===========================
    def load_dataset(self):
        """
        Loads and returns (X, y, encoder) consistently.
        NOTE: Scaling is now handled in run_pipeline.py after train-test split
        to prevent data leakage.
        """
        cache_file = os.path.join(self.cache_dir, f"features_{self.dataset_name}.npz")
        cache_encoder = os.path.join(self.cache_dir, f"encoder_{self.dataset_name}.joblib")

        # Try cached version
        if os.path.exists(cache_file) and os.path.exists(cache_encoder):
            print(f"[INFO] Loading cached dataset {self.dataset_name}...")
            try:
                cached = np.load(cache_file, allow_pickle=True)
                X, y = cached["X"], cached["y"]
                encoder = joblib.load(cache_encoder)
                print(f"[INFO] Loaded from cache: X{X.shape}, y{y.shape}")
                return X, y, encoder
            except Exception as e:
                print(f"[WARNING] Cache loading failed: {e}. Regenerating...")

        # Load fresh data
        data_format = self.dataset_config.get("format", "tabular")
        if data_format == "tabular":
            X, y = self._load_tabular()
        elif data_format == "raw":
            X, y = self._load_raw()
        else:
            raise ValueError(f"Unsupported dataset format: {data_format}")

        # Encode labels
        encoder = LabelEncoder()
        y = encoder.fit_transform(y)
        print(f"[INFO] Label classes: {encoder.classes_}")

        # Extract features (if enabled) - For RAW data only
        if self.apply_feature_extraction and data_format == "raw":
            print("[INFO] Extracting features from raw EEG...")
            sampling_rate = self.dataset_config.get("sampling_rate", 128)
            X = extract_features(
                eeg_data=X,
                dataset_format=data_format,
                fs=sampling_rate,
                include_bandpower=self.include_bandpower,
                include_stft=self.include_stft,
                include_entropy=self.include_entropy,      # <-- NEW
                include_wavelet=self.include_wavelet,      # <-- NEW
                normalize=self.per_channel_norm            # <-- NEW
            )
            print(f"[INFO] Feature extraction complete: X{X.shape}")

        # Cache safely
        try:
            np.savez_compressed(cache_file, X=X, y=y)
            joblib.dump(encoder, cache_encoder)
            print(f"[INFO] Cached dataset → {cache_file}, {cache_encoder}")
        except Exception as e:
            print(f"[WARNING] Failed to cache dataset: {e}")

        return X, y, encoder

    # ===========================
    # TABULAR LOADER
    # ===========================
    def _load_tabular(self):
        path = self.dataset_config["path"]
        if not os.path.exists(path):
            raise FileNotFoundError(f"[ERROR] CSV not found: {path}")

        try:
            # First, try reading with headers
            df = pd.read_csv(path)
            channels = self.dataset_config["channels"]
            label_column = self.dataset_config["label_column"]
            
            # Check if expected columns exist
            if not all(col in df.columns for col in channels):
                print(f"[WARNING] Expected column names not found. Reading CSV without headers...")
                # Read without headers, skipping first row (which is the actual header)
                df = pd.read_csv(path, header=None, skiprows=1, low_memory=False)
                
                # Last column is the label
                num_features = len(df.columns) - 1
                print(f"[INFO] Detected {num_features} feature columns + 1 label column")
                
                # Use all feature columns (0 to num_features-1)
                X = df.iloc[:, :num_features].values
                y = df.iloc[:, num_features].values  # Last column
                
                print(f"[INFO] Tabular dataset loaded (skipped header) → {path}")
                print(f"[INFO] Shape: X={X.shape}, y={y.shape}")
                print(f"[INFO] Using all {num_features} feature columns")
            else:
                # Columns exist, use them
                if not channels:
                    raise ValueError("No channels specified in config for tabular data")
                
                X = df[channels].values
                y = df[label_column].values
                
                print(f"[INFO] Tabular dataset loaded → {path}")
                print(f"[INFO] Shape: X={X.shape}, y={y.shape}")
                print(f"[INFO] Features: {len(channels)} columns")
            
            return X, y
        except Exception as e:
            raise RuntimeError(f"Failed to load tabular data from {path}: {e}")

    # ===========================
    # RAW EEG LOADER
    # ===========================
    def _load_raw(self):
        dataset_name = self.dataset_name.lower()
        path = self.dataset_config["path"]

        if not os.path.exists(path):
            raise FileNotFoundError(f"[ERROR] Dataset path not found: {path}")

        if dataset_name == "deap":
            return self._load_deap(path)
        elif dataset_name == "seed":
            return self._load_seed(path)
        else:
            raise ValueError(f"[ERROR] Raw loader not implemented for {dataset_name}")

    def _load_deap(self, folder_path):
        """Load DEAP dataset from .mat files"""
        print("[INFO] Loading DEAP dataset...")
        all_data, all_labels = [], []
        
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"DEAP folder not found: {folder_path}")
            
        mat_files = [f for f in os.listdir(folder_path) if f.endswith('.mat')]
        if not mat_files:
            raise FileNotFoundError(f"No .mat files found in {folder_path}")
            
        print(f"[INFO] Found {len(mat_files)} .mat files")
        
        for file in mat_files:
            try:
                data = scipy.io.loadmat(os.path.join(folder_path, file))
                eeg = data.get("data")
                labels = data.get("labels")
                
                if eeg is None or labels is None:
                    print(f"[WARNING] Missing data in {file}")
                    continue
                    
                # Create binary labels based on valence (configurable threshold)
                threshold = self.dataset_config.get("binary_threshold", 5)
                valence = (labels[:, 0] > threshold).astype(int)
                
                all_data.append(eeg)
                all_labels.append(valence)
                print(f"[INFO] Loaded {file}: {eeg.shape}, {valence.shape}")
                
            except Exception as e:
                print(f"[WARNING] Failed to load {file}: {e}")
                continue

        if not all_data:
            raise RuntimeError("No valid data loaded from DEAP dataset")
            
        X = np.concatenate(all_data, axis=0)
        y = np.concatenate(all_labels, axis=0)
        
        print(f"[INFO] DEAP dataset loaded: X{X.shape}, y{y.shape}")
        return X, y

    def _load_seed(self, folder_path):
        """Load SEED dataset from .mat files"""
        print("[INFO] Loading SEED dataset...")
        all_data, all_labels = [], []
        
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"SEED folder not found: {folder_path}")
            
        mat_files = [f for f in os.listdir(folder_path) if f.endswith('.mat')]
        if not mat_files:
            raise FileNotFoundError(f"No .mat files found in {folder_path}")
            
        print(f"[INFO] Found {len(mat_files)} .mat files")
        
        for file in mat_files:
            try:
                data = scipy.io.loadmat(os.path.join(folder_path, file))
                eeg = data.get("EEG")
                labels = data.get("label")
                
                if eeg is None or labels is None:
                    print(f"[WARNING] Missing data in {file}")
                    continue
                    
                all_data.append(eeg)
                all_labels.append(labels.flatten())
                print(f"[INFO] Loaded {file}: {eeg.shape}, {labels.shape}")
                
            except Exception as e:
                print(f"[WARNING] Failed to load {file}: {e}")
                continue

        if not all_data:
            raise RuntimeError("No valid data loaded from SEED dataset")
            
        X = np.concatenate(all_data, axis=0)
        y = np.concatenate(all_labels, axis=0)
        
        print(f"[INFO] SEED dataset loaded: X{X.shape}, y{y.shape}")
        return X, y

    # ===========================
    # CROSS DATASET
    # ===========================
    def get_cross_dataset(self):
        if not self.cross_dataset_enabled:
            raise RuntimeError("Cross-dataset mode is not enabled in config")

        train_name = self.config["cross_dataset"]["train"]
        test_name = self.config["cross_dataset"]["test"]

        print(f"[INFO] Cross-dataset mode → Train: {train_name}, Test: {test_name}")

        # Load training dataset
        self.dataset_name = train_name
        self.dataset_config = self.config["datasets"][train_name]
        X_train, y_train, enc_train = self.load_dataset()

        # Load testing dataset
        self.dataset_name = test_name
        self.dataset_config = self.config["datasets"][test_name]
        X_test, y_test, enc_test = self.load_dataset()

        print(f"[INFO] Cross-dataset loaded: Train{X_train.shape}, Test{X_test.shape}")
        return X_train, y_train, enc_train, X_test, y_test, enc_test
    