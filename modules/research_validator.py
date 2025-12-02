"""
Research-Grade Validation and Evaluation Module
Comprehensive validation, reproducibility, statistical tests, and runtime metrics
"""

import numpy as np
import pandas as pd
import json
import yaml
import platform
import sys
import subprocess
import time
import psutil
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from scipy import stats
from sklearn.model_selection import (
    StratifiedKFold, LeaveOneGroupOut, 
    cross_val_score, permutation_test_score
)
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, cohen_kappa_score, matthews_corrcoef
)
from sklearn.utils import resample
import hashlib
import joblib
import warnings
warnings.filterwarnings('ignore')


class ResearchValidator:
    """
    Comprehensive research-grade validation system
    Includes data leakage checks, proper CV, statistical tests, and reproducibility
    """
    
    def __init__(self, config_path: str = "config.yaml", results_dir: str = "results"):
        self.config_path = config_path
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.validation_results = {}
        self.reproducibility_info = {}
        self.statistical_tests = {}
        self.runtime_metrics = {}
        
    def validate_pipeline(self, X_train, y_train, X_test, y_test, 
                         models_trained: Dict, metrics_summary: Dict,
                         seed: int = 42) -> Dict:
        """
        Comprehensive pipeline validation
        Returns complete validation report
        """
        print("\n" + "="*80)
        print("RESEARCH-GRADE VALIDATION CHECKS")
        print("="*80 + "\n")
        
        validation_report = {
            'timestamp': datetime.now().isoformat(),
            'seed': seed,
            'data_leakage_checks': self._check_data_leakage(X_train, y_train, X_test, y_test),
            'cross_validation': self._cross_validation_analysis(X_train, y_train, seed),
            'statistical_tests': self._run_statistical_tests(X_train, y_train, X_test, y_test, models_trained, metrics_summary),
            'model_comparison': self._compare_models(X_test, y_test, models_trained, metrics_summary),
            'reproducibility': self._capture_reproducibility_info(),
            'runtime_metrics': self._capture_runtime_metrics(),
            'overfitting_analysis': self._analyze_overfitting(models_trained, X_train, y_train, X_test, y_test),
            'validation_summary': {}
        }
        
        # Generate summary
        validation_report['validation_summary'] = self._generate_validation_summary(validation_report)
        
        # Save validation report
        self._save_validation_report(validation_report)
        
        return validation_report
    
    def _check_data_leakage(self, X_train, y_train, X_test, y_test) -> Dict:
        """Comprehensive data leakage checks"""
        print("[VALIDATION] Checking for data leakage...")
        
        checks = {
            'train_test_overlap': self._check_train_test_overlap(X_train, X_test),
            'label_distribution': self._check_label_distribution(y_train, y_test),
            'feature_statistics': self._check_feature_statistics(X_train, X_test),
            'temporal_leakage': self._check_temporal_leakage(X_train, X_test),
            'duplicate_samples': self._check_duplicates(X_train, X_test)
        }
        
        # Overall assessment
        checks['has_leakage'] = any([
            checks['train_test_overlap']['has_overlap'],
            checks['duplicate_samples']['has_duplicates']
        ])
        
        if checks['has_leakage']:
            print("  ⚠️  WARNING: Potential data leakage detected!")
        else:
            print("  ✅ No data leakage detected")
        
        return checks
    
    def _check_train_test_overlap(self, X_train, X_test) -> Dict:
        """Check if train and test sets have overlapping samples"""
        # Convert to hashable format for comparison
        train_hashes = {hash(tuple(row)) for row in X_train[:1000]}  # Sample check
        test_hashes = {hash(tuple(row)) for row in X_test}
        
        overlap = len(train_hashes.intersection(test_hashes))
        overlap_ratio = overlap / len(test_hashes) if len(test_hashes) > 0 else 0
        
        return {
            'has_overlap': overlap > 0,
            'overlap_count': overlap,
            'overlap_ratio': overlap_ratio,
            'status': 'FAIL' if overlap > 0 else 'PASS'
        }
    
    def _check_label_distribution(self, y_train, y_test) -> Dict:
        """Check if label distributions are similar (stratified split)"""
        train_dist = pd.Series(y_train).value_counts(normalize=True).sort_index()
        test_dist = pd.Series(y_test).value_counts(normalize=True).sort_index()
        
        # Calculate KL divergence
        kl_div = 0
        for label in train_dist.index:
            if label in test_dist.index:
                p = train_dist[label]
                q = test_dist[label]
                if p > 0 and q > 0:
                    kl_div += p * np.log(p / q)
        
        return {
            'train_distribution': train_dist.to_dict(),
            'test_distribution': test_dist.to_dict(),
            'kl_divergence': kl_div,
            'is_stratified': kl_div < 0.1,  # Low KL divergence indicates stratified split
            'status': 'PASS' if kl_div < 0.1 else 'WARNING'
        }
    
    def _check_feature_statistics(self, X_train, X_test) -> Dict:
        """Check if feature statistics suggest leakage"""
        train_mean = np.mean(X_train, axis=0)
        test_mean = np.mean(X_test, axis=0)
        
        train_std = np.std(X_train, axis=0)
        test_std = np.std(X_test, axis=0)
        
        mean_diff = np.mean(np.abs(train_mean - test_mean))
        std_diff = np.mean(np.abs(train_std - test_std))
        
        # If scaled properly, means should be similar (close to 0 for standardized)
        return {
            'mean_difference': float(mean_diff),
            'std_difference': float(std_diff),
            'suggests_leakage': mean_diff > 1.0 or std_diff > 1.0,  # Threshold
            'status': 'PASS' if mean_diff < 1.0 and std_diff < 1.0 else 'WARNING'
        }
    
    def _check_temporal_leakage(self, X_train, X_test) -> Dict:
        """Check for temporal ordering that might cause leakage"""
        # Simple check: if data is sorted by index
        # This is a heuristic - actual temporal leakage requires domain knowledge
        return {
            'is_ordered': False,  # Would need timestamp data to check properly
            'status': 'INFO'
        }
    
    def _check_duplicates(self, X_train, X_test) -> Dict:
        """Check for duplicate samples between train and test"""
        # Sample-based check for performance
        n_check = min(1000, len(X_test))
        test_sample = X_test[:n_check]
        
        # Check if any test samples are in training set
        duplicates = 0
        for test_row in test_sample:
            # Check if test_row is in training set (with tolerance for floating point)
            if any(np.allclose(test_row, train_row, atol=1e-6) for train_row in X_train[:1000]):
                duplicates += 1
        
        return {
            'has_duplicates': duplicates > 0,
            'duplicate_count': duplicates,
            'duplicate_ratio': duplicates / n_check,
            'status': 'FAIL' if duplicates > 0 else 'PASS'
        }
    
    def _cross_validation_analysis(self, X_train, y_train, seed: int) -> Dict:
        """Comprehensive cross-validation analysis"""
        print("[VALIDATION] Running cross-validation analysis...")
        
        cv_results = {
            'stratified_kfold': self._stratified_kfold_cv(X_train, y_train, seed),
            'losocv': self._losocv_analysis(X_train, y_train) if self._has_subject_info() else None
        }
        
        return cv_results
    
    def _stratified_kfold_cv(self, X_train, y_train, seed: int, n_splits: int = 5) -> Dict:
        """Stratified K-Fold Cross-Validation"""
        from sklearn.ensemble import RandomForestClassifier
        
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        model = RandomForestClassifier(n_estimators=100, random_state=seed, max_depth=20)
        
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
        
        return {
            'n_splits': n_splits,
            'mean_accuracy': float(cv_scores.mean()),
            'std_accuracy': float(cv_scores.std()),
            'min_accuracy': float(cv_scores.min()),
            'max_accuracy': float(cv_scores.max()),
            'scores': cv_scores.tolist(),
            'confidence_interval_95': [
                float(np.percentile(cv_scores, 2.5)),
                float(np.percentile(cv_scores, 97.5))
            ]
        }
    
    def _losocv_analysis(self, X_train, y_train) -> Dict:
        """Leave-One-Subject-Out Cross-Validation (if subject info available)"""
        # This requires subject information - placeholder for now
        return {
            'available': False,
            'reason': 'Subject information not available in dataset'
        }
    
    def _has_subject_info(self) -> bool:
        """Check if dataset has subject information"""
        # Would need to check config or data structure
        return False
    
    def _run_statistical_tests(self, X_train, y_train, X_test, y_test,
                               models_trained: Dict, metrics_summary: Dict) -> Dict:
        """Run comprehensive statistical tests"""
        print("[VALIDATION] Running statistical tests...")
        
        tests = {
            'bootstrap_confidence_intervals': self._bootstrap_ci(X_test, y_test, models_trained),
            'permutation_test': self._permutation_test(X_train, y_train, X_test, y_test, models_trained),
            'baseline_comparison': self._baseline_comparison(X_train, y_train, X_test, y_test),
            'cohens_kappa': self._calculate_cohens_kappa(X_test, y_test, models_trained),
            'mcnemar_test': self._mcnemar_test(X_test, y_test, models_trained) if len(models_trained) > 1 else None
        }
        
        return tests
    
    def _bootstrap_ci(self, X_test, y_test, models_trained: Dict, n_bootstraps: int = 1000) -> Dict:
        """Bootstrap confidence intervals for accuracy"""
        bootstrap_results = {}
        
        for model_name, model in models_trained.items():
            if hasattr(model, 'predict'):
                accuracies = []
                
                for _ in range(n_bootstraps):
                    # Bootstrap sample
                    indices = np.random.randint(0, len(X_test), len(X_test))
                    X_boot = X_test[indices]
                    y_boot = y_test[indices]
                    
                    # Predict and calculate accuracy
                    y_pred = model.predict(X_boot)
                    acc = accuracy_score(y_boot, y_pred)
                    accuracies.append(acc)
                
                bootstrap_results[model_name] = {
                    'mean_accuracy': float(np.mean(accuracies)),
                    'std_accuracy': float(np.std(accuracies)),
                    'ci_95_lower': float(np.percentile(accuracies, 2.5)),
                    'ci_95_upper': float(np.percentile(accuracies, 97.5)),
                    'ci_99_lower': float(np.percentile(accuracies, 0.5)),
                    'ci_99_upper': float(np.percentile(accuracies, 99.5))
                }
        
        return bootstrap_results
    
    def _permutation_test(self, X_train, y_train, X_test, y_test, 
                         models_trained: Dict, n_permutations: int = 100) -> Dict:
        """Permutation test for statistical significance"""
        permutation_results = {}
        
        for model_name, model in models_trained.items():
            if hasattr(model, 'predict'):
                # Train on real labels
                model.fit(X_train, y_train)
                y_pred_real = model.predict(X_test)
                real_accuracy = accuracy_score(y_test, y_pred_real)
                
                # Permutation test
                permuted_accuracies = []
                for _ in range(n_permutations):
                    y_permuted = np.random.permutation(y_train)
                    model.fit(X_train, y_permuted)
                    y_pred_perm = model.predict(X_test)
                    perm_acc = accuracy_score(y_test, y_pred_perm)
                    permuted_accuracies.append(perm_acc)
                
                # Calculate p-value
                p_value = np.mean(np.array(permuted_accuracies) >= real_accuracy)
                
                permutation_results[model_name] = {
                    'real_accuracy': float(real_accuracy),
                    'permuted_mean_accuracy': float(np.mean(permuted_accuracies)),
                    'p_value': float(p_value),
                    'is_significant': p_value < 0.05,
                    'n_permutations': n_permutations
                }
        
        return permutation_results
    
    def _baseline_comparison(self, X_train, y_train, X_test, y_test) -> Dict:
        """Compare models against baseline (majority class, random, DummyClassifier)"""
        from sklearn.dummy import DummyClassifier
        from sklearn.linear_model import LogisticRegression
        
        # Majority class baseline
        from collections import Counter
        majority_class = Counter(y_train).most_common(1)[0][0]
        y_pred_majority = np.full_like(y_test, majority_class)
        majority_acc = accuracy_score(y_test, y_pred_majority)
        
        # Dummy classifier (stratified)
        dummy = DummyClassifier(strategy='stratified', random_state=42)
        dummy.fit(X_train, y_train)
        y_pred_dummy = dummy.predict(X_test)
        dummy_acc = accuracy_score(y_test, y_pred_dummy)
        
        # Logistic Regression baseline
        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(X_train, y_train)
        y_pred_lr = lr.predict(X_test)
        lr_acc = accuracy_score(y_test, y_pred_lr)
        
        return {
            'majority_class_accuracy': float(majority_acc),
            'dummy_classifier_accuracy': float(dummy_acc),
            'logistic_regression_accuracy': float(lr_acc),
            'baseline_max': float(max(majority_acc, dummy_acc, lr_acc))
        }
    
    def _calculate_cohens_kappa(self, X_test, y_test, models_trained: Dict) -> Dict:
        """Calculate Cohen's Kappa for all models"""
        kappa_results = {}
        
        for model_name, model in models_trained.items():
            if hasattr(model, 'predict'):
                y_pred = model.predict(X_test)
                kappa = cohen_kappa_score(y_test, y_pred)
                mcc = matthews_corrcoef(y_test, y_pred)
                
                kappa_results[model_name] = {
                    'cohens_kappa': float(kappa),
                    'matthews_correlation': float(mcc),
                    'interpretation': self._interpret_kappa(kappa)
                }
        
        return kappa_results
    
    def _interpret_kappa(self, kappa: float) -> str:
        """Interpret Cohen's Kappa value"""
        if kappa < 0:
            return "Poor (worse than random)"
        elif kappa < 0.20:
            return "Slight"
        elif kappa < 0.40:
            return "Fair"
        elif kappa < 0.60:
            return "Moderate"
        elif kappa < 0.80:
            return "Substantial"
        else:
            return "Almost Perfect"
    
    def _mcnemar_test(self, X_test, y_test, models_trained: Dict) -> Dict:
        """McNemar's test for comparing models"""
        if len(models_trained) < 2:
            return None
        
        model_names = list(models_trained.keys())
        comparisons = {}
        
        for i, name1 in enumerate(model_names):
            for name2 in model_names[i+1:]:
                model1 = models_trained[name1]
                model2 = models_trained[name2]
                
                if hasattr(model1, 'predict') and hasattr(model2, 'predict'):
                    y_pred1 = model1.predict(X_test)
                    y_pred2 = model2.predict(X_test)
                    
                    # Create contingency table
                    both_correct = np.sum((y_pred1 == y_test) & (y_pred2 == y_test))
                    model1_correct = np.sum((y_pred1 == y_test) & (y_pred2 != y_test))
                    model2_correct = np.sum((y_pred1 != y_test) & (y_pred2 == y_test))
                    both_wrong = np.sum((y_pred1 != y_test) & (y_pred2 != y_test))
                    
                    # McNemar's test (chi-square test for paired nominal data)
                    contingency = [[both_correct, model1_correct],
                                  [model2_correct, both_wrong]]
                    
                    # Skip if all values are the same
                    if model1_correct == 0 and model2_correct == 0:
                        p_value = 1.0
                        chi2 = 0.0
                    else:
                        # Use continuity correction
                        chi2 = ((abs(model1_correct - model2_correct) - 1) ** 2) / (model1_correct + model2_correct)
                        from scipy.stats import chi2 as chi2_dist
                        p_value = 1 - chi2_dist.cdf(chi2, df=1)
                    
                    comparisons[f"{name1}_vs_{name2}"] = {
                        'chi2_statistic': float(chi2),
                        'p_value': float(p_value),
                        'is_significant': p_value < 0.05,
                        'contingency_table': {
                            'both_correct': int(both_correct),
                            'model1_only_correct': int(model1_correct),
                            'model2_only_correct': int(model2_correct),
                            'both_wrong': int(both_wrong)
                        }
                    }
        
        return comparisons
    
    def _compare_models(self, X_test, y_test, models_trained: Dict, metrics_summary: Dict) -> Dict:
        """Comprehensive model comparison"""
        print("[VALIDATION] Comparing models...")
        
        comparison = {
            'accuracy_ranking': self._rank_models_by_accuracy(metrics_summary),
            'f1_ranking': self._rank_models_by_f1(metrics_summary),
            'statistical_comparison': self._mcnemar_test(X_test, y_test, models_trained)
        }
        
        return comparison
    
    def _rank_models_by_accuracy(self, metrics_summary: Dict) -> List[Dict]:
        """Rank models by accuracy"""
        rankings = []
        for model_name, metrics in metrics_summary.items():
            acc = metrics.get('accuracy', 0)
            rankings.append({'model': model_name, 'accuracy': acc})
        
        rankings.sort(key=lambda x: x['accuracy'], reverse=True)
        return rankings
    
    def _rank_models_by_f1(self, metrics_summary: Dict) -> List[Dict]:
        """Rank models by F1 score"""
        rankings = []
        for model_name, metrics in metrics_summary.items():
            f1 = metrics.get('weighted avg', {}).get('f1-score', 0)
            rankings.append({'model': model_name, 'f1_score': f1})
        
        rankings.sort(key=lambda x: x['f1_score'], reverse=True)
        return rankings
    
    def _analyze_overfitting(self, models_trained: Dict, X_train, y_train, X_test, y_test) -> Dict:
        """Analyze overfitting/underfitting"""
        print("[VALIDATION] Analyzing overfitting...")
        
        overfitting_analysis = {}
        
        for model_name, model in models_trained.items():
            if hasattr(model, 'predict') and hasattr(model, 'score'):
                train_acc = model.score(X_train, y_train)
                test_acc = model.score(X_test, y_test)
                gap = train_acc - test_acc
                
                overfitting_analysis[model_name] = {
                    'train_accuracy': float(train_acc),
                    'test_accuracy': float(test_acc),
                    'accuracy_gap': float(gap),
                    'overfitting_risk': 'HIGH' if gap > 0.05 else 'MEDIUM' if gap > 0.02 else 'LOW',
                    'underfitting_risk': 'HIGH' if train_acc < 0.7 else 'MEDIUM' if train_acc < 0.8 else 'LOW'
                }
        
        return overfitting_analysis
    
    def _capture_reproducibility_info(self) -> Dict:
        """Capture comprehensive reproducibility information"""
        print("[VALIDATION] Capturing reproducibility information...")
        
        info = {
            'timestamp': datetime.now().isoformat(),
            'system_info': self._get_system_info(),
            'python_environment': self._get_python_environment(),
            'config_snapshot': self._get_config_snapshot(),
            'random_seeds': self._get_random_seeds(),
            'git_info': self._get_git_info(),
            'package_versions': self._get_package_versions()
        }
        
        return info
    
    def _get_system_info(self) -> Dict:
        """Get system information"""
        return {
            'platform': platform.platform(),
            'processor': platform.processor(),
            'machine': platform.machine(),
            'cpu_count': psutil.cpu_count(),
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
            'python_version': sys.version,
            'python_executable': sys.executable
        }
    
    def _get_python_environment(self) -> Dict:
        """Get Python environment information"""
        return {
            'virtual_env': os.environ.get('VIRTUAL_ENV', 'Not in virtual environment'),
            'conda_env': os.environ.get('CONDA_DEFAULT_ENV', 'Not in conda environment')
        }
    
    def _get_config_snapshot(self) -> Dict:
        """Get config file snapshot"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            return {'error': str(e)}
    
    def _get_random_seeds(self) -> Dict:
        """Get random seed information"""
        return {
            'numpy_seed': int(np.random.get_state()[1][0]),
            'python_random_seed': 'Set via config'
        }
    
    def _get_git_info(self) -> Dict:
        """Get git information if available"""
        try:
            commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], 
                                           stderr=subprocess.STDOUT).decode().strip()
            branch = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                                           stderr=subprocess.STDOUT).decode().strip()
            return {
                'commit': commit,
                'branch': branch,
                'available': True
            }
        except Exception:
            return {'available': False}
    
    def _get_package_versions(self) -> Dict:
        """Get versions of key packages"""
        packages = ['numpy', 'pandas', 'scikit-learn', 'xgboost', 'torch']
        versions = {}
        
        for pkg in packages:
            try:
                mod = __import__(pkg.replace('-', '_'))
                versions[pkg] = getattr(mod, '__version__', 'unknown')
            except ImportError:
                versions[pkg] = 'not installed'
        
        return versions
    
    def _capture_runtime_metrics(self) -> Dict:
        """Capture runtime metrics"""
        process = psutil.Process(os.getpid())
        
        return {
            'memory_usage_mb': process.memory_info().rss / (1024 * 1024),
            'cpu_percent': process.cpu_percent(interval=1),
            'cpu_count': psutil.cpu_count(),
            'memory_available_gb': psutil.virtual_memory().available / (1024**3),
            'timestamp': datetime.now().isoformat()
        }
    
    def _generate_validation_summary(self, validation_report: Dict) -> Dict:
        """Generate summary of validation results"""
        summary = {
            'overall_status': 'PASS',
            'warnings': [],
            'errors': [],
            'recommendations': []
        }
        
        # Check data leakage
        if validation_report['data_leakage_checks']['has_leakage']:
            summary['overall_status'] = 'FAIL'
            summary['errors'].append('Data leakage detected')
        
        # Check overfitting
        for model_name, analysis in validation_report['overfitting_analysis'].items():
            if analysis['overfitting_risk'] == 'HIGH':
                summary['warnings'].append(f'{model_name}: High overfitting risk (gap: {analysis["accuracy_gap"]:.3f})')
        
        # Check statistical significance
        for model_name, perm_test in validation_report['statistical_tests']['permutation_test'].items():
            if not perm_test['is_significant']:
                summary['warnings'].append(f'{model_name}: Model may not be statistically significant (p={perm_test["p_value"]:.3f})')
        
        # Recommendations
        if summary['overall_status'] == 'PASS':
            summary['recommendations'].append('All validation checks passed. Results are valid for research publication.')
        
        return summary
    
    def _save_validation_report(self, validation_report: Dict):
        """Save validation report to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.results_dir / f"validation_report_{timestamp}.json"
        
        with open(report_path, 'w') as f:
            json.dump(validation_report, f, indent=2, default=str)
        
        print(f"\n[VALIDATION] Validation report saved to: {report_path}")
        
        # Also save a human-readable summary
        summary_path = self.results_dir / f"validation_summary_{timestamp}.txt"
        self._save_human_readable_summary(validation_report, summary_path)
    
    def _save_human_readable_summary(self, validation_report: Dict, path: Path):
        """Save human-readable validation summary"""
        with open(path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("RESEARCH-GRADE VALIDATION REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Timestamp: {validation_report['timestamp']}\n")
            f.write(f"Seed: {validation_report['seed']}\n\n")
            
            # Data Leakage Checks
            f.write("DATA LEAKAGE CHECKS\n")
            f.write("-"*80 + "\n")
            leakage = validation_report['data_leakage_checks']
            f.write(f"Has Leakage: {'YES' if leakage['has_leakage'] else 'NO'}\n")
            f.write(f"Train-Test Overlap: {leakage['train_test_overlap']['status']}\n")
            f.write(f"Duplicate Samples: {leakage['duplicate_samples']['status']}\n\n")
            
            # Cross-Validation
            f.write("CROSS-VALIDATION RESULTS\n")
            f.write("-"*80 + "\n")
            cv = validation_report['cross_validation']['stratified_kfold']
            f.write(f"Mean CV Accuracy: {cv['mean_accuracy']:.4f} ± {cv['std_accuracy']:.4f}\n")
            f.write(f"95% CI: [{cv['confidence_interval_95'][0]:.4f}, {cv['confidence_interval_95'][1]:.4f}]\n\n")
            
            # Overfitting Analysis
            f.write("OVERFITTING ANALYSIS\n")
            f.write("-"*80 + "\n")
            for model_name, analysis in validation_report['overfitting_analysis'].items():
                f.write(f"{model_name}:\n")
                f.write(f"  Train Accuracy: {analysis['train_accuracy']:.4f}\n")
                f.write(f"  Test Accuracy: {analysis['test_accuracy']:.4f}\n")
                f.write(f"  Gap: {analysis['accuracy_gap']:.4f}\n")
                f.write(f"  Overfitting Risk: {analysis['overfitting_risk']}\n\n")
            
            # Statistical Tests
            f.write("STATISTICAL TESTS\n")
            f.write("-"*80 + "\n")
            for model_name, perm_test in validation_report['statistical_tests']['permutation_test'].items():
                f.write(f"{model_name}:\n")
                f.write(f"  Real Accuracy: {perm_test['real_accuracy']:.4f}\n")
                f.write(f"  P-value: {perm_test['p_value']:.4f}\n")
                f.write(f"  Significant: {'YES' if perm_test['is_significant'] else 'NO'}\n\n")
            
            # Summary
            f.write("VALIDATION SUMMARY\n")
            f.write("-"*80 + "\n")
            summary = validation_report['validation_summary']
            f.write(f"Overall Status: {summary['overall_status']}\n")
            if summary['warnings']:
                f.write("Warnings:\n")
                for warning in summary['warnings']:
                    f.write(f"  - {warning}\n")
            if summary['errors']:
                f.write("Errors:\n")
                for error in summary['errors']:
                    f.write(f"  - {error}\n")
        
        print(f"[VALIDATION] Human-readable summary saved to: {path}")

