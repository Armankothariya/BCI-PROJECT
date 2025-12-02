"""Gather all raw outputs and logs from the latest runs"""
import json
import joblib
import pandas as pd
import numpy as np
import timeit
from sklearn.metrics import confusion_matrix

print("="*70)
print("📄 GATHERING LATEST RUN LOGS AND OUTPUTS")
print("="*70)

# --- 1. Feature Extraction Results ---
print("\n--- 1. Feature Extraction Results (5x5 Sample) ---")
try:
    df = pd.read_csv('results/X_test_processed.csv')
    print(df.iloc[:5, :5])
except Exception as e:
    print(f"Error: {e}")

# --- 2. Model Training Logs ---
print("\n--- 2. Model Training Logs (RandomForest) ---")
try:
    with open('results/model_summary_20251016_141205.json') as f:
        data = json.load(f)
    print(json.dumps(data['metrics']['RandomForest'], indent=2))
except Exception as e:
    print(f"Error: {e}")

# --- 3. Confusion Matrix ---
print("\n--- 3. Confusion Matrix ---")
try:
    bundle = joblib.load('models/production_model.joblib')
    X = pd.read_csv('results/X_test_processed.csv').values
    y_numeric = pd.read_csv('results/y_test.csv').iloc[:, 0].values
    
    # Convert numeric y to string labels for comparison
    y_strings = bundle['label_encoder'].inverse_transform(y_numeric)
    
    p_numeric = bundle['model'].predict(X)
    p_strings = bundle['label_encoder'].inverse_transform(p_numeric)
    
    cm = confusion_matrix(y_strings, p_strings, labels=bundle['label_encoder'].classes_)
    
    print(f"Labels: {bundle['label_encoder'].classes_}")
    print("Matrix:")
    print(cm)
except Exception as e:
    print(f"Error: {e}")

# --- 4. Latency Log ---
print("\n--- 4. Latency Log ---")
try:
    bundle = joblib.load('models/production_model.joblib')
    X_sample = pd.read_csv('results/X_test_processed.csv').values[0].reshape(1, -1)
    
    # Define the statement to be timed
    stmt = lambda: bundle['model'].predict(X_sample)
    
    # Time the execution
    times = timeit.repeat(stmt, repeat=10, number=100)
    avg_latency = (sum(times) / (10 * 100)) * 1000
    
    print(f"Average Latency: {avg_latency:.2f} ms")
    print("(Based on 10 runs of 100 predictions each)")
except Exception as e:
    print(f"Error: {e}")

print("\n" + "="*70)
print("✅ LOGS GATHERED SUCCESSFULLY!")
print("="*70)
