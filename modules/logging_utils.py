import sqlite3
import json
import h5py
import pandas as pd
import numpy as np
from datetime import datetime
import os

class BCILogger:
    def __init__(self, log_dir="results"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.db_path = os.path.join(log_dir, "predictions.db")
        self._init_db()
        
    def _init_db(self):
        """Initialize SQLite database for predictions"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS predictions
                     (timestamp TEXT, session_id TEXT, emotion TEXT, 
                      confidence REAL, raw_eeg BLOB, features BLOB)''')
        conn.commit()
        conn.close()
    
    def log_prediction(self, emotion, confidence, raw_eeg=None, features=None, session_id="default"):
        """Log a prediction to SQLite database"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Convert arrays to bytes for storage
        raw_eeg_blob = raw_eeg.tobytes() if raw_eeg is not None else None
        features_blob = features.tobytes() if features is not None else None
        
        c.execute("INSERT INTO predictions VALUES (?, ?, ?, ?, ?, ?)",
                  (datetime.now().isoformat(), session_id, emotion, confidence, 
                   raw_eeg_blob, features_blob))
        conn.commit()
        conn.close()
    
    def log_to_hdf5(self, session_data, session_id=None):
        """Log complete session data to HDF5 file"""
        if session_id is None:
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        h5_path = os.path.join(self.log_dir, f"session_{session_id}.h5")
        
        with h5py.File(h5_path, 'w') as hf:
            # Store metadata
            metadata = {
                "session_id": session_id,
                "timestamp": datetime.now().isoformat(),
                "data_format": "eeg_predictions"
            }
            hf.attrs['metadata'] = json.dumps(metadata)
            
            # Store data arrays
            for key, value in session_data.items():
                if isinstance(value, np.ndarray):
                    hf.create_dataset(key, data=value)
                else:
                    hf.create_dataset(key, data=np.array(value))
        
        return h5_path
    
    def get_session_history(self, session_id="default", limit=100):
        """Retrieve prediction history for a session"""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query(
            f"SELECT timestamp, emotion, confidence FROM predictions WHERE session_id = ? ORDER BY timestamp DESC LIMIT {limit}",
            conn, params=[session_id])
        conn.close()
        return df