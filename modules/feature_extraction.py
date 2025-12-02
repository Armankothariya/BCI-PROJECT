# ============================================
# FEATURE EXTRACTION MODULE FOR EEG DATA (RESEARCH-GRADE)
# - REMOVED: sample_entropy function that returned random values
# - FIXED: Only includes valid, non-random feature extraction methods
# ============================================

import os
import hashlib
import numpy as np
from scipy.stats import skew, kurtosis
from scipy.signal import welch, stft
import pywt

# -------------------------
# GLOBAL CACHE DIRECTORY
# -------------------------
CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# -------------------------
# 1) HELPER: Normalization
# -------------------------
def _normalize_channel(signal):
    """Apply Z-score normalization to a single EEG channel."""
    signal = np.array(signal, dtype=np.float64)
    if signal.size == 0 or np.all(np.isnan(signal)):
        return np.zeros_like(signal)
    return (signal - np.nanmean(signal)) / (np.nanstd(signal) + 1e-6)

# -------------------------
# 2) STATISTICAL FEATURES
# -------------------------
def extract_extended_stats_features(eeg_data):
    """Compute mean, var, skew, kurtosis, median, range, std per channel."""
    eeg_data = np.nan_to_num(np.array(eeg_data, dtype=np.float64))
    means = np.mean(eeg_data, axis=1)
    variances = np.var(eeg_data, axis=1)
    skews = np.nan_to_num(skew(eeg_data, axis=1))
    kurtoses = np.nan_to_num(kurtosis(eeg_data, axis=1))
    medians = np.median(eeg_data, axis=1)
    ranges = np.ptp(eeg_data, axis=1)
    stds = np.std(eeg_data, axis=1)
    return np.vstack([means, variances, skews, kurtoses, medians, ranges, stds]).T

# -------------------------
# 3) BANDPOWER FEATURES
# -------------------------
def compute_bandpower(signal, fs, band):
    """Compute average power in a given EEG band using Welch's method."""
    signal = np.nan_to_num(signal)
    if signal.size < 4:  # too short
        return 0.0
    freqs, psd = welch(signal, fs=fs, nperseg=min(256, len(signal)))
    idx_band = np.logical_and(freqs >= band[0], freqs <= band[1])
    if not np.any(idx_band):
        return 0.0
    return np.trapz(psd[idx_band], freqs[idx_band])

def extract_bandpower_features(eeg_data, fs, normalize=True):
    """Delta, Theta, Alpha, Beta, Gamma bandpower per channel."""
    bands = {
        "delta": (1, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta": (13, 30),
        "gamma": (30, 45)
    }
    features = []
    for row in eeg_data:
        row = _normalize_channel(row) if normalize else np.nan_to_num(row)
        bp = [compute_bandpower(row, fs, band) for band in bands.values()]
        features.append(bp)
    return np.array(features)

# -------------------------
# 4) STFT FEATURES
# -------------------------
def extract_stft_features(signal, fs, max_bins=50):
    """Mean STFT magnitude across time, truncated/padded to max_bins."""
    signal = np.nan_to_num(signal)
    if signal.size < 8:
        return [0.0] * max_bins
    f, _, Zxx = stft(signal, fs=fs, nperseg=min(128, len(signal)))
    magnitude = np.abs(Zxx)
    feat = np.mean(magnitude, axis=1)
    # pad/truncate to fixed size
    if len(feat) < max_bins:
        feat = np.pad(feat, (0, max_bins - len(feat)))
    else:
        feat = feat[:max_bins]
    return feat.tolist()

# -------------------------
# 5) ENTROPY FEATURES (ONLY PERMUTATION ENTROPY)
# -------------------------
def permutation_entropy(x, order=3, delay=1):
    """
    Calculate Permutation Entropy of a signal.
    A measure of complexity based on comparing neighboring values.
    """
    x = np.nan_to_num(np.array(x))
    n = len(x)
    if n < order * delay:
        return 0.0
    
    # Create overlapping sequences of length 'order' with spacing 'delay'
    permutations = np.array([x[i:i+order*delay:delay] for i in range(n - order*delay + 1)])
    if permutations.size == 0:
        return 0.0
    
    # Get the ordinal pattern for each sequence
    ranks = np.argsort(permutations, axis=1)
    
    # Count frequency of each unique pattern
    unique_patterns, counts = np.unique(ranks, axis=0, return_counts=True)
    if len(counts) == 0:
        return 0.0
        
    # Calculate probability distribution
    probs = counts / counts.sum()
    
    # Calculate permutation entropy
    return -np.sum(probs * np.log2(probs + 1e-12))

# -------------------------
# 6) WAVELET FEATURES
# -------------------------
def wavelet_energy(sig, wavelet='db4', level=4):
    """Wavelet energy features."""
    sig = np.nan_to_num(sig)
    try:
        coeffs = pywt.wavedec(sig, wavelet, level=level)
        return [np.sum(np.square(c)) for c in coeffs]
    except Exception:
        return [0.0] * (level + 1)

# -------------------------
# 7) MASTER FEATURE EXTRACTOR
# -------------------------
def extract_features(eeg_data,
                     dataset_format,
                     fs=128,
                     include_bandpower=True,
                     include_stft=False,
                     include_entropy=False,
                     include_wavelet=False,
                     normalize=True):
    """
    Main robust feature extractor for EEG data.
    Returns a 2D array of shape (n_samples, n_features)
    """
    eeg_data = np.nan_to_num(np.array(eeg_data, dtype=np.float64))

    # Statistical features (always included)
    features = extract_extended_stats_features(eeg_data)
    print(f"[FEATURE EXTRACTION] Statistical features: {features.shape[1]} dimensions")

    # Bandpower features
    if include_bandpower:
        bp_features = extract_bandpower_features(eeg_data, fs, normalize)
        features = np.hstack([features, bp_features])
        print(f"[FEATURE EXTRACTION] + Bandpower features: {bp_features.shape[1]} dimensions")

    # STFT features
    if include_stft:
        stft_features = [extract_stft_features(row, fs) for row in eeg_data]
        stft_features = np.array(stft_features)
        features = np.hstack([features, stft_features])
        print(f"[FEATURE EXTRACTION] + STFT features: {stft_features.shape[1]} dimensions")

    # Entropy features (Only permutation entropy - sample entropy was removed)
    if include_entropy:
        entropy_features = []
        for row in eeg_data:
            ch_feats = [permutation_entropy(row)]  # Only permutation entropy
            entropy_features.append(ch_feats)
        entropy_features = np.array(entropy_features)
        features = np.hstack([features, entropy_features])
        print(f"[FEATURE EXTRACTION] + Entropy features: {entropy_features.shape[1]} dimensions")

    # Wavelet features
    if include_wavelet:
        wavelet_features = []
        for row in eeg_data:
            ch_feats = wavelet_energy(row)
            wavelet_features.append(ch_feats)
        wavelet_features = np.array(wavelet_features)
        features = np.hstack([features, wavelet_features])
        print(f"[FEATURE EXTRACTION] + Wavelet features: {wavelet_features.shape[1]} dimensions")

    print(f"[FEATURE EXTRACTION] Total features: {features.shape[1]} dimensions")
    return features

# -------------------------
# 8) CACHING WRAPPER
# -------------------------
def extract_and_cache(eeg_data, cache_name, **kwargs):
    """Extract features with cache. Hashes kwargs to avoid mixing configs."""
    # Build hash for settings
    settings_str = str(kwargs)
    settings_hash = hashlib.md5(settings_str.encode()).hexdigest()[:8]
    cache_path = os.path.join(CACHE_DIR, f"{cache_name}_{settings_hash}.npy")

    if os.path.exists(cache_path):
        print(f"[INFO] Loading cached features from {cache_path}")
        return np.load(cache_path, allow_pickle=True)

    print("[INFO] Extracting features (this may take time)...")
    features = extract_features(eeg_data, **kwargs)
    np.save(cache_path, features)
    print(f"[INFO] Features cached at {cache_path}")
    return features