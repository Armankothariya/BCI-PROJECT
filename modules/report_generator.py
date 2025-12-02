# modules/report_generator.py
# ============================================
# 📋 RESEARCH-GRADE REPORT GENERATOR
# Reproducibility bundles, statistical reports, and exports
# ============================================

import os
import json
import zipfile
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

from modules.utils import get_git_info, bootstrap_confidence_interval

logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Generate comprehensive research reports with reproducibility bundles.
    """
    
    def __init__(self, output_dir: str = "results"):
        """
        Initialize report generator.
        
        Args:
            output_dir: Directory for saving reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        logger.info(f"ReportGenerator initialized: output_dir={output_dir}")
    
    def generate_model_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str,
        metadata: Optional[Dict] = None
    ) -> Dict:
        """
        Generate comprehensive model performance report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Name of the model
            metadata: Additional metadata
        
        Returns:
            dict: Report data
        """
        logger.info(f"Generating report for {model_name}")
        
        # Calculate metrics
        report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        accuracy = accuracy_score(y_true, y_pred)
        conf_matrix = confusion_matrix(y_true, y_pred)
        
        # Calculate confidence intervals
        ci_accuracy = bootstrap_confidence_interval(
            y_true, y_pred, 
            accuracy_score,
            n_bootstraps=1000
        )
        
        # Compile report
        report = {
            'model_name': model_name,
            'timestamp': datetime.now().isoformat(),
            'metrics': {
                'accuracy': accuracy,
                'accuracy_ci': ci_accuracy,
                'classification_report': report_dict,
                'confusion_matrix': conf_matrix.tolist()
            },
            'metadata': metadata or {},
            'git_info': get_git_info()
        }
        
        # Save report
        report_path = self.output_dir / f"{model_name.lower()}_report_{self.timestamp}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Report saved: {report_path}")
        
        return report
    
    def create_reproducibility_bundle(
        self,
        models_dir: str = "models",
        config_files: Optional[List[str]] = None,
        include_data: bool = False
    ) -> Path:
        """
        Create complete reproducibility bundle.
        
        Args:
            models_dir: Directory containing models
            config_files: List of config files to include
            include_data: Whether to include processed data
        
        Returns:
            Path: Path to created bundle
        """
        logger.info("Creating reproducibility bundle")
        
        bundle_name = f"senito_reproducibility_{self.timestamp}.zip"
        bundle_path = self.output_dir / bundle_name
        
        with zipfile.ZipFile(bundle_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add models
            models_path = Path(models_dir)
            if models_path.exists():
                for model_file in models_path.glob("*.joblib"):
                    zipf.write(model_file, f"models/{model_file.name}")
                logger.info(f"Added {len(list(models_path.glob('*.joblib')))} model files")
            
            # Add results
            if self.output_dir.exists():
                for result_file in self.output_dir.glob("*"):
                    if result_file.is_file() and result_file.suffix in ['.json', '.csv', '.png']:
                        zipf.write(result_file, f"results/{result_file.name}")
                logger.info(f"Added result files")
            
            # Add configuration files
            if config_files is None:
                config_files = ["config.yaml", "requirements.txt", "music_map.yaml"]
            
            for config_file in config_files:
                config_path = Path(config_file)
                if config_path.exists():
                    zipf.write(config_path, config_path.name)
            
            # Add processed data (optional)
            if include_data:
                data_files = [
                    "results/X_train_processed.csv",
                    "results/X_test_processed.csv",
                    "results/y_test.csv",
                    "results/feature_names.csv"
                ]
                for data_file in data_files:
                    data_path = Path(data_file)
                    if data_path.exists():
                        zipf.write(data_path, f"data/{data_path.name}")
            
            # Create experiment metadata
            metadata = self._create_experiment_metadata()
            metadata_json = json.dumps(metadata, indent=2)
            zipf.writestr("experiment_metadata.json", metadata_json)
            
            # Create README
            readme = self._create_bundle_readme()
            zipf.writestr("README.md", readme)
        
        logger.info(f"Reproducibility bundle created: {bundle_path} ({bundle_path.stat().st_size / 1024:.1f} KB)")
        
        return bundle_path
    
    def _create_experiment_metadata(self) -> Dict:
        """Create comprehensive experiment metadata"""
        import sys
        import platform
        
        try:
            import sklearn
            import numpy as np
            import pandas as pd
            import streamlit as st
            
            packages = {
                'scikit-learn': sklearn.__version__,
                'numpy': np.__version__,
                'pandas': pd.__version__,
                'streamlit': st.__version__
            }
        except:
            packages = {}
        
        metadata = {
            'experiment': {
                'name': 'Senito BCI Emotion Classification',
                'timestamp': datetime.now().isoformat(),
                'version': '2.0'
            },
            'environment': {
                'python_version': sys.version,
                'platform': platform.platform(),
                'architecture': platform.machine(),
                'packages': packages
            },
            'reproducibility': {
                'random_seed': 42,
                'git_info': get_git_info(),
                'bundle_created': datetime.now().isoformat()
            }
        }
        
        return metadata
    
    def _create_bundle_readme(self) -> str:
        """Create README for reproducibility bundle"""
        readme = f"""# Senito BCI Reproducibility Bundle

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Contents

- `models/` - Trained ML models (.joblib format)
- `results/` - Metrics, reports, and visualizations
- `data/` - Processed features (if included)
- `experiment_metadata.json` - Complete experiment metadata
- Configuration files (config.yaml, requirements.txt, etc.)

## Reproduction Instructions

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv senito_env
source senito_env/bin/activate  # On Windows: senito_env\\Scripts\\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Load Model

```python
import joblib
model_bundle = joblib.load('models/production_model.joblib')
```

### 3. Run Inference

```python
from modules.emotion_system import EmotionSystem
import numpy as np

# Initialize system
system = EmotionSystem(model_bundle)

# Process features
result = system.process(features)
print(f"Emotion: {{result['emotion']}} ({{result['confidence']:.2%}})")
```

### 4. Review Results

See `results/` directory for:
- Model performance metrics (.json)
- Confusion matrices (.png)
- Classification reports (.csv)
- Feature importances

## Citation

If you use this work, please cite:

```
@misc{{senito2024,
  title={{Senito: Emotion-Aware Brain-Computer Interface}},
  year={{2024}},
  note={{Reproducibility bundle}}
}}
```

## Contact & Support

For questions or issues, please refer to the project documentation.

---

*This bundle ensures complete reproducibility of experimental results.*
"""
        return readme
    
    def generate_comparison_report(
        self,
        models_results: Dict[str, Dict],
        output_format: str = "html"
    ) -> Path:
        """
        Generate model comparison report.
        
        Args:
            models_results: Dict mapping model names to their results
            output_format: Output format (html, pdf, or markdown)
        
        Returns:
            Path: Path to generated report
        """
        logger.info(f"Generating comparison report ({output_format})")
        
        # Create comparison DataFrame
        comparison_data = []
        for model_name, results in models_results.items():
            metrics = results.get('metrics', {})
            comparison_data.append({
                'Model': model_name,
                'Accuracy': metrics.get('accuracy', 0),
                'Precision': metrics.get('weighted avg', {}).get('precision', 0),
                'Recall': metrics.get('weighted avg', {}).get('recall', 0),
                'F1-Score': metrics.get('weighted avg', {}).get('f1-score', 0)
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('Accuracy', ascending=False)
        
        # Generate report based on format
        if output_format == "html":
            report_path = self._generate_html_report(comparison_df, models_results)
        elif output_format == "markdown":
            report_path = self._generate_markdown_report(comparison_df, models_results)
        else:
            report_path = self.output_dir / f"comparison_{self.timestamp}.csv"
            comparison_df.to_csv(report_path, index=False)
        
        logger.info(f"Comparison report saved: {report_path}")
        
        return report_path
    
    def _generate_html_report(self, comparison_df: pd.DataFrame, models_results: Dict) -> Path:
        """Generate HTML comparison report"""
        report_path = self.output_dir / f"comparison_{self.timestamp}.html"
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Senito Model Comparison Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
                .container {{ background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                h1 {{ color: #2E86AB; }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #2E86AB; color: white; }}
                tr:hover {{ background-color: #f5f5f5; }}
                .metric {{ font-weight: bold; color: #2E86AB; }}
                .timestamp {{ color: #666; font-size: 0.9em; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🧠 Senito Model Comparison Report</h1>
                <p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                
                <h2>Model Performance Comparison</h2>
                {comparison_df.to_html(index=False, classes='table')}
                
                <h2>Best Model</h2>
                <p>The best performing model is <span class="metric">{comparison_df.iloc[0]['Model']}</span> 
                with an accuracy of <span class="metric">{comparison_df.iloc[0]['Accuracy']:.2%}</span>.</p>
            </div>
        </body>
        </html>
        """
        
        with open(report_path, 'w') as f:
            f.write(html)
        
        return report_path
    
    def _generate_markdown_report(self, comparison_df: pd.DataFrame, models_results: Dict) -> Path:
        """Generate Markdown comparison report"""
        report_path = self.output_dir / f"comparison_{self.timestamp}.md"
        
        markdown = f"""# Senito Model Comparison Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Model Performance Comparison

{comparison_df.to_markdown(index=False)}

## Best Model

The best performing model is **{comparison_df.iloc[0]['Model']}** with an accuracy of **{comparison_df.iloc[0]['Accuracy']:.2%}**.

## Methodology

- Dataset split: 80/20 train/test
- Cross-validation: 3-fold
- Metrics: Accuracy, Precision, Recall, F1-Score
- Confidence intervals calculated via bootstrap (1000 iterations)

## Reproducibility

This report is part of a reproducibility bundle. See `experiment_metadata.json` for complete details.
"""
        
        with open(report_path, 'w') as f:
            f.write(markdown)
        
        return report_path
    
    def export_session_data(
        self,
        session_history: List[Dict],
        metadata: Optional[Dict] = None
    ) -> Path:
        """
        Export session data with metadata.
        
        Args:
            session_history: List of prediction dictionaries
            metadata: Optional session metadata
        
        Returns:
            Path: Path to exported file
        """
        export_path = self.output_dir / f"session_export_{self.timestamp}.json"
        
        export_data = {
            'session_info': {
                'timestamp': datetime.now().isoformat(),
                'total_predictions': len(session_history),
                'metadata': metadata or {}
            },
            'history': session_history
        }
        
        with open(export_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Session data exported: {export_path}")
        
        return export_path
    
    def create_performance_plots(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str,
        class_names: Optional[List[str]] = None
    ) -> List[Path]:
        """
        Create performance visualization plots.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Name of model
            class_names: Optional class names
        
        Returns:
            list: Paths to created plots
        """
        logger.info(f"Creating performance plots for {model_name}")
        
        plots = []
        
        # Confusion matrix
        cm_path = self.output_dir / f"{model_name.lower()}_confusion_matrix_{self.timestamp}.png"
        self._plot_confusion_matrix(y_true, y_pred, cm_path, class_names)
        plots.append(cm_path)
        
        logger.info(f"Created {len(plots)} plots")
        
        return plots
    
    def _plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        output_path: Path,
        class_names: Optional[List[str]] = None
    ):
        """Plot and save confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names or np.unique(y_true),
            yticklabels=class_names or np.unique(y_true)
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
    
    def generate_session_pdf(
        self,
        session_data: Dict[str, Any],
        prediction_history: List[Dict],
        output_filename: Optional[str] = None
    ) -> Path:
        """
        Generate comprehensive PDF report for a session.
        
        Args:
            session_data: Session metadata and statistics
            prediction_history: List of prediction dictionaries
            output_filename: Optional custom filename
        
        Returns:
            Path: Path to generated PDF
        """
        if not REPORTLAB_AVAILABLE:
            logger.error("ReportLab not available. Cannot generate PDF.")
            raise ImportError("ReportLab is required for PDF generation. Install with: pip install reportlab")
        
        logger.info("Generating session PDF report")
        
        # Create PDF filename
        if output_filename is None:
            output_filename = f"senito_session_report_{self.timestamp}.pdf"
        
        pdf_path = self.output_dir / output_filename
        
        # Create PDF document
        doc = SimpleDocTemplate(
            str(pdf_path),
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18
        )
        
        # Container for PDF elements
        elements = []
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#2E86AB'),
            spaceAfter=30,
            alignment=TA_CENTER
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#2E86AB'),
            spaceAfter=12,
            spaceBefore=12
        )
        
        # Title
        elements.append(Paragraph("🧠 Senito BCI Session Report", title_style))
        elements.append(Spacer(1, 0.2 * inch))
        
        # Session Information
        elements.append(Paragraph("Session Information", heading_style))
        session_info_data = [
            ['Session ID', session_data.get('session_id', 'N/A')],
            ['Date', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            ['Duration', session_data.get('duration', 'N/A')],
            ['Total Predictions', str(len(prediction_history))],
            ['System Mode', session_data.get('system_mode', 'N/A')],
        ]
        
        session_table = Table(session_info_data, colWidths=[2.5*inch, 4*inch])
        session_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#E8F4F8')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ]))
        elements.append(session_table)
        elements.append(Spacer(1, 0.3 * inch))
        
        # Prediction Summary
        if prediction_history:
            elements.append(Paragraph("Prediction Summary", heading_style))
            
            # Calculate statistics
            valid_preds = [p for p in prediction_history if not p.get('error', False)]
            
            if valid_preds:
                emotions = [p.get('emotion', 'Unknown') for p in valid_preds]
                confidences = [p.get('confidence', 0) for p in valid_preds]
                
                emotion_counts = {}
                for emotion in emotions:
                    emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
                
                summary_data = [
                    ['Metric', 'Value'],
                    ['Valid Predictions', str(len(valid_preds))],
                    ['Failed Predictions', str(len(prediction_history) - len(valid_preds))],
                    ['Average Confidence', f"{np.mean(confidences):.2%}"],
                    ['Max Confidence', f"{np.max(confidences):.2%}"],
                    ['Min Confidence', f"{np.min(confidences):.2%}"],
                ]
                
                summary_table = Table(summary_data, colWidths=[3*inch, 3.5*inch])
                summary_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2E86AB')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 12),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                ]))
                elements.append(summary_table)
                elements.append(Spacer(1, 0.2 * inch))
                
                # Emotion Distribution
                elements.append(Paragraph("Emotion Distribution", heading_style))
                emotion_dist_data = [['Emotion', 'Count', 'Percentage']]
                for emotion, count in sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True):
                    percentage = (count / len(valid_preds)) * 100
                    emotion_dist_data.append([emotion.title(), str(count), f"{percentage:.1f}%"])
                
                emotion_table = Table(emotion_dist_data, colWidths=[2.5*inch, 2*inch, 2*inch])
                emotion_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2E86AB')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 11),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
                    ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                ]))
                elements.append(emotion_table)
            
            elements.append(Spacer(1, 0.3 * inch))
            
            # Recent Predictions (last 10)
            elements.append(PageBreak())
            elements.append(Paragraph("Recent Predictions", heading_style))
            
            recent_preds = prediction_history[-min(20, len(prediction_history)):][::-1]
            pred_data = [['#', 'Timestamp', 'Emotion', 'Confidence', 'Status']]
            
            for idx, pred in enumerate(recent_preds, 1):
                timestamp = pred.get('timestamp', 'N/A')
                if isinstance(timestamp, str):
                    # Try to format timestamp if it's a string
                    try:
                        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                        timestamp = dt.strftime('%H:%M:%S')
                    except:
                        pass
                
                emotion = pred.get('emotion', 'N/A')
                confidence = pred.get('confidence', 0)
                status = '✓ Success' if not pred.get('error', False) else '✗ Failed'
                
                pred_data.append([
                    str(idx),
                    str(timestamp),
                    emotion.title() if isinstance(emotion, str) else 'N/A',
                    f"{confidence:.1%}" if isinstance(confidence, (int, float)) else 'N/A',
                    status
                ])
            
            pred_table = Table(pred_data, colWidths=[0.5*inch, 1.3*inch, 1.8*inch, 1.5*inch, 1.4*inch])
            pred_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2E86AB')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (0, -1), 'CENTER'),
                ('ALIGN', (1, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ]))
            elements.append(pred_table)
        
        # Footer
        elements.append(Spacer(1, 0.5 * inch))
        footer_style = ParagraphStyle(
            'Footer',
            parent=styles['Normal'],
            fontSize=8,
            textColor=colors.grey,
            alignment=TA_CENTER
        )
        elements.append(Paragraph(
            f"Generated by Senito BCI v2.0 | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            footer_style
        ))
        elements.append(Paragraph("🧠 Emotion-Aware Brain-Computer Interface", footer_style))
        
        # Build PDF
        doc.build(elements)
        
        logger.info(f"PDF report generated: {pdf_path} ({pdf_path.stat().st_size / 1024:.1f} KB)")
        
        return pdf_path
