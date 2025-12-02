"""
Enhanced Visualizations Module
Displays all pipeline results, creates comprehensive reports
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json
from datetime import datetime

def display_all_pipeline_images(results_dir):
    """
    Display all images from the latest pipeline run
    """
    results_dir = Path(results_dir)
    
    st.header("📊 Complete Pipeline Results")
    
    # Get all PNG files
    png_files = list(results_dir.glob("*.png"))
    
    if not png_files:
        st.warning("⚠️ No pipeline result images found. Run the pipeline first!")
        st.code("python run_pipeline.py", language="bash")
        return
    
    # Get latest timestamp
    timestamps = []
    for f in png_files:
        parts = f.stem.split('_')
        if len(parts) >= 2 and parts[-1].isdigit():
            timestamps.append(parts[-1])
    
    if timestamps:
        latest_timestamp = max(set(timestamps))
        st.info(f"📅 Showing results from: {latest_timestamp}")
    else:
        latest_timestamp = None
    
    # Section 1: Model Confusion Matrices
    st.subheader("🎯 Confusion Matrices - All Models")
    
    models = ['randomforest', 'xgboost', 'svm']
    cols = st.columns(3)
    
    for idx, model in enumerate(models):
        with cols[idx]:
            cm_path = results_dir / f"{model}_confusion_matrix.png"
            if cm_path.exists():
                st.image(str(cm_path), caption=f"{model.upper()}", use_column_width=True)
            else:
                st.warning(f"No {model} matrix found")
    
    st.divider()
    
    # Section 2: Performance Comparisons
    st.subheader("📈 Model Performance Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if latest_timestamp:
            acc_path = results_dir / f"accuracy_comparison_{latest_timestamp}.png"
            if acc_path.exists():
                st.image(str(acc_path), caption="Accuracy Comparison", use_column_width=True)
    
    with col2:
        if latest_timestamp:
            f1_path = results_dir / f"f1_comparison_{latest_timestamp}.png"
            if f1_path.exists():
                st.image(str(f1_path), caption="F1-Score Comparison", use_column_width=True)
    
    st.divider()
    
    # Section 3: Feature Importance
    st.subheader("🔍 Feature Importance Analysis")
    
    for model in ['randomforest', 'xgboost']:
        csv_path = results_dir / f"{model}_feature_importances.csv"
        if csv_path.exists():
            st.markdown(f"### {model.upper()} - Top 20 Features")
            
            df = pd.read_csv(csv_path).head(20)
            
            # Create horizontal bar chart
            fig = px.bar(
                df, 
                x='importance', 
                y='feature',
                orientation='h',
                title=f"Top 20 Most Important Features - {model.upper()}",
                labels={'importance': 'Importance Score', 'feature': 'Feature Name'},
                color='importance',
                color_continuous_scale='Viridis'
            )
            
            fig.update_layout(
                height=600,
                yaxis={'categoryorder': 'total ascending'},
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Section 4: Metrics Summary
    st.subheader("📊 Detailed Metrics Summary")
    
    metrics_data = []
    for model in models:
        metrics_path = results_dir / f"{model}_metrics.json"
        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
                metrics_data.append({
                    'Model': model.upper(),
                    'Accuracy': f"{metrics.get('accuracy', 0):.4f}",
                    'Precision': f"{metrics.get('macro avg', {}).get('precision', 0):.4f}",
                    'Recall': f"{metrics.get('macro avg', {}).get('recall', 0):.4f}",
                    'F1-Score': f"{metrics.get('macro avg', {}).get('f1-score', 0):.4f}"
                })
    
    if metrics_data:
        metrics_df = pd.DataFrame(metrics_data)
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
        
        # Create comparison chart
        metrics_melted = pd.DataFrame(metrics_data).melt(
            id_vars='Model', 
            var_name='Metric', 
            value_name='Score'
        )
        metrics_melted['Score'] = metrics_melted['Score'].astype(float)
        
        fig = px.bar(
            metrics_melted,
            x='Model',
            y='Score',
            color='Metric',
            barmode='group',
            title="Model Performance Metrics Comparison",
            labels={'Score': 'Score Value', 'Model': 'Model Type'}
        )
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)


def display_model_comparison_dashboard(results_dir):
    """
    Create comprehensive model comparison dashboard
    """
    results_dir = Path(results_dir)
    
    st.header("🏆 Model Comparison Dashboard")
    
    models = ['randomforest', 'xgboost', 'svm']
    
    # Load all metrics
    all_metrics = {}
    for model in models:
        metrics_path = results_dir / f"{model}_metrics.json"
        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                all_metrics[model] = json.load(f)
    
    if not all_metrics:
        st.warning("No model metrics found!")
        return
    
    # Create comparison table
    comparison_data = []
    for model, metrics in all_metrics.items():
        comparison_data.append({
            'Model': model.upper(),
            'Accuracy': metrics.get('accuracy', 0),
            'Precision (Macro)': metrics.get('macro avg', {}).get('precision', 0),
            'Recall (Macro)': metrics.get('macro avg', {}).get('recall', 0),
            'F1-Score (Macro)': metrics.get('macro avg', {}).get('f1-score', 0),
            'Support': metrics.get('macro avg', {}).get('support', 0)
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    best_accuracy = comparison_df.loc[comparison_df['Accuracy'].idxmax()]
    best_f1 = comparison_df.loc[comparison_df['F1-Score (Macro)'].idxmax()]
    best_precision = comparison_df.loc[comparison_df['Precision (Macro)'].idxmax()]
    best_recall = comparison_df.loc[comparison_df['Recall (Macro)'].idxmax()]
    
    with col1:
        st.metric(
            "🎯 Best Accuracy",
            f"{best_accuracy['Accuracy']:.2%}",
            best_accuracy['Model']
        )
    
    with col2:
        st.metric(
            "⚖️ Best F1-Score",
            f"{best_f1['F1-Score (Macro)']:.2%}",
            best_f1['Model']
        )
    
    with col3:
        st.metric(
            "🎪 Best Precision",
            f"{best_precision['Precision (Macro)']:.2%}",
            best_precision['Model']
        )
    
    with col4:
        st.metric(
            "📡 Best Recall",
            f"{best_recall['Recall (Macro)']:.2%}",
            best_recall['Model']
        )
    
    st.divider()
    
    # Detailed comparison table
    st.subheader("📋 Detailed Comparison")
    st.dataframe(
        comparison_df.style.highlight_max(axis=0, subset=['Accuracy', 'Precision (Macro)', 'Recall (Macro)', 'F1-Score (Macro)']),
        use_container_width=True,
        hide_index=True
    )
    
    # Radar chart comparison
    st.subheader("🕸️ Performance Radar Chart")
    
    fig = go.Figure()
    
    metrics_to_plot = ['Accuracy', 'Precision (Macro)', 'Recall (Macro)', 'F1-Score (Macro)']
    
    for _, row in comparison_df.iterrows():
        fig.add_trace(go.Scatterpolar(
            r=[row[m] for m in metrics_to_plot],
            theta=metrics_to_plot,
            fill='toself',
            name=row['Model']
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0.9, 1.0]  # Adjust based on your metrics
            )
        ),
        showlegend=True,
        title="Model Performance Comparison (Radar)"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Per-class performance
    st.subheader("📊 Per-Class Performance")
    
    for model, metrics in all_metrics.items():
        with st.expander(f"📈 {model.upper()} - Class-wise Metrics"):
            class_data = []
            for class_id in ['0', '1', '2']:
                if class_id in metrics:
                    class_data.append({
                        'Class': f"Class {class_id}",
                        'Precision': metrics[class_id].get('precision', 0),
                        'Recall': metrics[class_id].get('recall', 0),
                        'F1-Score': metrics[class_id].get('f1-score', 0),
                        'Support': int(metrics[class_id].get('support', 0))
                    })
            
            if class_data:
                class_df = pd.DataFrame(class_data)
                st.dataframe(class_df, use_container_width=True, hide_index=True)
                
                # Bar chart for class performance
                fig = px.bar(
                    class_df.melt(id_vars=['Class', 'Support'], var_name='Metric', value_name='Score'),
                    x='Class',
                    y='Score',
                    color='Metric',
                    barmode='group',
                    title=f"{model.upper()} - Per-Class Performance"
                )
                st.plotly_chart(fig, use_container_width=True)


def export_session_data(prediction_history, format='csv'):
    """
    Export session data in various formats
    """
    if not prediction_history:
        st.warning("No session data to export!")
        return None
    
    # Convert to DataFrame
    df = pd.DataFrame(list(prediction_history))
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if format == 'csv':
        filename = f"session_export_{timestamp}.csv"
        df.to_csv(filename, index=False)
        return filename
    
    elif format == 'json':
        filename = f"session_export_{timestamp}.json"
        df.to_json(filename, orient='records', indent=2)
        return filename
    
    elif format == 'excel':
        filename = f"session_export_{timestamp}.xlsx"
        df.to_excel(filename, index=False, engine='openpyxl')
        return filename
    
    return None


def create_session_analytics(prediction_history):
    """
    Create comprehensive session analytics
    """
    if not prediction_history:
        st.warning("No session data available!")
        return
    
    st.header("📊 Session Analytics")
    
    df = pd.DataFrame(list(prediction_history))
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Predictions", len(df))
    
    with col2:
        avg_confidence = df['confidence'].mean()
        st.metric("Avg Confidence", f"{avg_confidence:.1%}")
    
    with col3:
        most_common = df['emotion'].mode()[0] if not df['emotion'].mode().empty else 'N/A'
        st.metric("Dominant Emotion", most_common.upper())
    
    with col4:
        stability = 1 - df['confidence'].std()
        st.metric("Stability Score", f"{stability:.1%}")
    
    st.divider()
    
    # Visualizations
    col_viz1, col_viz2 = st.columns(2)
    
    with col_viz1:
        # Emotion distribution
        st.subheader("🎭 Emotion Distribution")
        emotion_counts = df['emotion'].value_counts()
        
        fig = px.pie(
            values=emotion_counts.values,
            names=emotion_counts.index,
            title="Emotion Distribution",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col_viz2:
        # Confidence distribution
        st.subheader("📊 Confidence Distribution")
        
        fig = px.histogram(
            df,
            x='confidence',
            nbins=30,
            title="Confidence Score Distribution",
            labels={'confidence': 'Confidence Score', 'count': 'Frequency'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Timeline
    st.subheader("📈 Emotion Timeline")
    
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        fig = px.line(
            df,
            x='timestamp',
            y='confidence',
            color='emotion',
            title="Emotion Confidence Over Time",
            labels={'confidence': 'Confidence', 'timestamp': 'Time'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Transition analysis
    st.subheader("🔄 Emotion Transitions")
    
    transitions = []
    for i in range(1, len(df)):
        if df.iloc[i]['emotion'] != df.iloc[i-1]['emotion']:
            transitions.append({
                'From': df.iloc[i-1]['emotion'],
                'To': df.iloc[i]['emotion'],
                'Time': df.iloc[i].get('timestamp', i)
            })
    
    if transitions:
        st.write(f"Total transitions: {len(transitions)}")
        
        # Transition matrix
        trans_df = pd.DataFrame(transitions)
        trans_matrix = pd.crosstab(trans_df['From'], trans_df['To'])
        
        fig = px.imshow(
            trans_matrix,
            labels=dict(x="To Emotion", y="From Emotion", color="Count"),
            title="Emotion Transition Matrix",
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No emotion transitions detected in this session")
    
    # Export options
    st.divider()
    st.subheader("💾 Export Session Data")
    
    col_exp1, col_exp2, col_exp3 = st.columns(3)
    
    with col_exp1:
        if st.button("📄 Export as CSV"):
            filename = export_session_data(prediction_history, 'csv')
            if filename:
                st.success(f"✅ Exported to {filename}")
    
    with col_exp2:
        if st.button("📋 Export as JSON"):
            filename = export_session_data(prediction_history, 'json')
            if filename:
                st.success(f"✅ Exported to {filename}")
    
    with col_exp3:
        if st.button("📊 Export as Excel"):
            try:
                filename = export_session_data(prediction_history, 'excel')
                if filename:
                    st.success(f"✅ Exported to {filename}")
            except ImportError:
                st.error("❌ Install openpyxl: pip install openpyxl")
