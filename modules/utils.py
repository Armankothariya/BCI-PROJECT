# modules/utils.py
# ============================================
# Utility Functions for Senito BCI
# ============================================

import os
import json
import logging
import zipfile
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import subprocess
import sys

import numpy as np

def setup_logging(log_dir: str = "results", level: str = "INFO"):
    """Setup comprehensive logging"""
    log_dir = Path(log_dir)
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / f"senito_{datetime.now().strftime('%Y%m%d')}.log"
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)

def get_git_info() -> Optional[Dict]:
    """Get git repository information for reproducibility"""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--git-dir'],
            capture_output=True,
            text=True,
            check=False
        )
        
        if result.returncode != 0:
            return None
        
        info = {}
        
        # Get commit hash
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True,
            text=True,
            check=True
        )
        info['commit'] = result.stdout.strip()
        
        # Get branch
        result = subprocess.run(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            capture_output=True,
            text=True,
            check=True
        )
        info['branch'] = result.stdout.strip()
        
        # Get status
        result = subprocess.run(
            ['git', 'status', '--porcelain'],
            capture_output=True,
            text=True,
            check=True
        )
        info['clean_working_dir'] = len(result.stdout.strip()) == 0
        
        return info
        
    except (subprocess.SubprocessError, FileNotFoundError):
        return None

def create_reproducibility_bundle() -> Optional[Path]:
    """Create reproducibility bundle ZIP file"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        bundle_name = f"senito_reproducibility_{timestamp}.zip"
        bundle_path = Path("results") / bundle_name
        
        with zipfile.ZipFile(bundle_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add models
            models_dir = Path("models")
            if models_dir.exists():
                for model_file in models_dir.glob("*.joblib"):
                    zipf.write(model_file, f"models/{model_file.name}")
            
            # Add results
            results_dir = Path("results")
            if results_dir.exists():
                for result_file in results_dir.glob("*"):
                    if result_file.is_file():
                        zipf.write(result_file, f"results/{result_file.name}")
            
            # Add configuration
            config_files = ["config.yaml", "requirements.txt", "DATASET_README.md"]
            for config_file in config_files:
                if Path(config_file).exists():
                    zipf.write(config_file, config_file)
            
            # Add experiment metadata
            metadata = {
                'timestamp': datetime.now().isoformat(),
                'git_info': get_git_info(),
                'python_version': sys.version,
                'created_by': 'Senito BCI Reproducibility System'
            }
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(metadata, f, indent=2)
                temp_path = Path(f.name)
            
            zipf.write(temp_path, "experiment_metadata.json")
            temp_path.unlink()
        
        logging.info(f"Created reproducibility bundle: {bundle_path}")
        return bundle_path
        
    except Exception as e:
        logging.error(f"Failed to create reproducibility bundle: {e}")
        return None

def generate_experiment_metadata(pipeline_config: Dict) -> Dict:
    """Generate comprehensive experiment metadata"""
    return {
        'experiment': {
            'timestamp': datetime.now().isoformat(),
            'pipeline_config': pipeline_config,
            'environment': {
                'python_version': sys.version,
                'platform': sys.platform,
                'working_directory': str(Path.cwd())
            },
            'git_info': get_git_info(),
            'reproducibility': {
                'bundle_created': False,
                'requirements_frozen': False
            }
        }
    }

def save_preprocessing_recipe(steps: Dict, output_path: Path):
    """Save preprocessing recipe for reproducibility"""
    recipe = {
        'timestamp': datetime.now().isoformat(),
        'preprocessing_steps': steps,
        'version': '1.0',
        'parameters': {
            'sampling_rate': 128,
            'epoch_length': 2.0,
            'notch_filter': 50.0,
            'bandpass_low': 1.0,
            'bandpass_high': 45.0
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(recipe, f, indent=2)

def bootstrap_confidence_interval(y_true, y_pred, metric_func, n_bootstraps=1000, confidence_level=0.95):
    """Calculate bootstrap confidence intervals for metrics"""
    bootstrapped_scores = []
    
    for _ in range(n_bootstraps):
        indices = np.random.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[indices])) < 2:
            continue
        
        score = metric_func(y_true[indices], y_pred[indices])
        bootstrapped_scores.append(score)
    
    alpha = (1 - confidence_level) / 2
    lower = np.percentile(bootstrapped_scores, 100 * alpha)
    upper = np.percentile(bootstrapped_scores, 100 * (1 - alpha))
    
    return {
        'mean': np.mean(bootstrapped_scores),
        'std': np.std(bootstrapped_scores),
        'confidence_interval': [lower, upper],
        'confidence_level': confidence_level
    }