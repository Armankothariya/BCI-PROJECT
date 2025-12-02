"""
3D Emotion Visualization
Interactive 3D scatter plots in valence-arousal-dominance space
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


class Emotion3DVisualizer:
    """
    Create interactive 3D visualizations of emotions
    """
    
    def __init__(self):
        self.emotion_colors = {
            'positive': '#2ecc71',
            'negative': '#e74c3c',
            'neutral': '#95a5a6'
        }
    
    def plot_3d_emotion_space(self, features, emotions, confidences=None, 
                              method='pca', title=None):
        """
        Plot emotions in 3D space using dimensionality reduction
        
        Args:
            features: Feature matrix (n_samples x n_features)
            emotions: Emotion labels
            confidences: Optional confidence scores
            method: 'pca' or 'tsne'
            title: Plot title
        
        Returns:
            plotly figure
        """
        # Reduce to 3D
        if method == 'pca':
            reducer = PCA(n_components=3)
            coords_3d = reducer.fit_transform(features)
            explained_var = reducer.explained_variance_ratio_
            axis_labels = [
                f'PC1 ({explained_var[0]:.1%})',
                f'PC2 ({explained_var[1]:.1%})',
                f'PC3 ({explained_var[2]:.1%})'
            ]
        elif method == 'tsne':
            reducer = TSNE(n_components=3, random_state=42)
            coords_3d = reducer.fit_transform(features)
            axis_labels = ['t-SNE 1', 't-SNE 2', 't-SNE 3']
        else:
            raise ValueError("Method must be 'pca' or 'tsne'")
        
        # Create DataFrame
        df = pd.DataFrame({
            'x': coords_3d[:, 0],
            'y': coords_3d[:, 1],
            'z': coords_3d[:, 2],
            'emotion': emotions,
            'confidence': confidences if confidences is not None else [1.0] * len(emotions)
        })
        
        # Create 3D scatter plot
        fig = go.Figure()
        
        for emotion in df['emotion'].unique():
            emotion_data = df[df['emotion'] == emotion]
            
            fig.add_trace(go.Scatter3d(
                x=emotion_data['x'],
                y=emotion_data['y'],
                z=emotion_data['z'],
                mode='markers',
                name=emotion.upper(),
                marker=dict(
                    size=8,
                    color=self.emotion_colors.get(emotion, 'gray'),
                    opacity=0.8,
                    line=dict(color='white', width=0.5)
                ),
                text=[f'{emotion.upper()}<br>Confidence: {c:.2%}' 
                      for c in emotion_data['confidence']],
                hovertemplate='<b>%{text}</b><br>' +
                             f'{axis_labels[0]}: %{{x:.2f}}<br>' +
                             f'{axis_labels[1]}: %{{y:.2f}}<br>' +
                             f'{axis_labels[2]}: %{{z:.2f}}<extra></extra>'
            ))
        
        # Update layout
        if title is None:
            title = f'3D Emotion Space ({method.upper()})'
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title=axis_labels[0],
                yaxis_title=axis_labels[1],
                zaxis_title=axis_labels[2],
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            height=700,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        return fig
    
    def plot_valence_arousal_dominance(self, valence, arousal, dominance, 
                                       emotions, confidences=None):
        """
        Plot emotions in VAD (Valence-Arousal-Dominance) space
        
        Args:
            valence: Valence scores
            arousal: Arousal scores
            dominance: Dominance scores
            emotions: Emotion labels
            confidences: Optional confidence scores
        
        Returns:
            plotly figure
        """
        df = pd.DataFrame({
            'Valence': valence,
            'Arousal': arousal,
            'Dominance': dominance,
            'Emotion': emotions,
            'Confidence': confidences if confidences is not None else [1.0] * len(emotions)
        })
        
        # Create 3D scatter
        fig = go.Figure()
        
        for emotion in df['Emotion'].unique():
            emotion_data = df[df['Emotion'] == emotion]
            
            fig.add_trace(go.Scatter3d(
                x=emotion_data['Valence'],
                y=emotion_data['Arousal'],
                z=emotion_data['Dominance'],
                mode='markers',
                name=emotion.upper(),
                marker=dict(
                    size=10,
                    color=self.emotion_colors.get(emotion, 'gray'),
                    opacity=0.7,
                    line=dict(color='white', width=1)
                ),
                text=[f'{emotion.upper()}<br>Conf: {c:.2%}' 
                      for c in emotion_data['Confidence']],
                hovertemplate='<b>%{text}</b><br>' +
                             'Valence: %{x:.2f}<br>' +
                             'Arousal: %{y:.2f}<br>' +
                             'Dominance: %{z:.2f}<extra></extra>'
            ))
        
        # Add reference planes
        fig.add_trace(go.Surface(
            x=[-1, 1],
            y=[-1, 1],
            z=[[0, 0], [0, 0]],
            opacity=0.1,
            showscale=False,
            colorscale=[[0, 'gray'], [1, 'gray']],
            name='Neutral Plane'
        ))
        
        fig.update_layout(
            title='Emotion Space: Valence-Arousal-Dominance (VAD)',
            scene=dict(
                xaxis_title='Valence (Negative ← → Positive)',
                yaxis_title='Arousal (Calm ← → Excited)',
                zaxis_title='Dominance (Submissive ← → Dominant)',
                xaxis=dict(range=[-1, 1]),
                yaxis=dict(range=[-1, 1]),
                zaxis=dict(range=[-1, 1]),
                camera=dict(
                    eye=dict(x=1.7, y=1.7, z=1.3)
                )
            ),
            height=700,
            showlegend=True
        )
        
        return fig
    
    def plot_emotion_trajectory_3d(self, features_timeline, emotions_timeline, 
                                   timestamps=None, method='pca'):
        """
        Plot emotion trajectory over time in 3D space
        
        Args:
            features_timeline: List of feature vectors over time
            emotions_timeline: List of emotions over time
            timestamps: Optional timestamps
            method: 'pca' or 'tsne'
        
        Returns:
            plotly figure
        """
        # Reduce to 3D
        if method == 'pca':
            reducer = PCA(n_components=3)
            coords_3d = reducer.fit_transform(features_timeline)
        else:
            reducer = TSNE(n_components=3, random_state=42)
            coords_3d = reducer.fit_transform(features_timeline)
        
        # Create figure
        fig = go.Figure()
        
        # Plot trajectory line
        fig.add_trace(go.Scatter3d(
            x=coords_3d[:, 0],
            y=coords_3d[:, 1],
            z=coords_3d[:, 2],
            mode='lines+markers',
            name='Trajectory',
            line=dict(color='gray', width=2),
            marker=dict(
                size=6,
                color=[self.emotion_colors.get(e, 'gray') for e in emotions_timeline],
                opacity=0.8
            ),
            text=[f'Time: {i}<br>Emotion: {e.upper()}' 
                  for i, e in enumerate(emotions_timeline)],
            hovertemplate='<b>%{text}</b><extra></extra>'
        ))
        
        # Add start and end markers
        fig.add_trace(go.Scatter3d(
            x=[coords_3d[0, 0]],
            y=[coords_3d[0, 1]],
            z=[coords_3d[0, 2]],
            mode='markers',
            name='Start',
            marker=dict(size=15, color='green', symbol='diamond'),
            hovertemplate='<b>START</b><extra></extra>'
        ))
        
        fig.add_trace(go.Scatter3d(
            x=[coords_3d[-1, 0]],
            y=[coords_3d[-1, 1]],
            z=[coords_3d[-1, 2]],
            mode='markers',
            name='End',
            marker=dict(size=15, color='red', symbol='diamond'),
            hovertemplate='<b>END</b><extra></extra>'
        ))
        
        fig.update_layout(
            title='Emotion Trajectory Over Time (3D)',
            scene=dict(
                xaxis_title='Dimension 1',
                yaxis_title='Dimension 2',
                zaxis_title='Dimension 3',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            height=700
        )
        
        return fig
    
    def plot_emotion_clusters_3d(self, features, emotions, show_centroids=True):
        """
        Plot emotion clusters in 3D with centroids
        
        Args:
            features: Feature matrix
            emotions: Emotion labels
            show_centroids: Whether to show cluster centroids
        
        Returns:
            plotly figure
        """
        # Reduce to 3D using PCA
        pca = PCA(n_components=3)
        coords_3d = pca.fit_transform(features)
        
        df = pd.DataFrame({
            'x': coords_3d[:, 0],
            'y': coords_3d[:, 1],
            'z': coords_3d[:, 2],
            'emotion': emotions
        })
        
        # Create figure
        fig = go.Figure()
        
        # Plot each emotion cluster
        for emotion in df['emotion'].unique():
            emotion_data = df[df['emotion'] == emotion]
            
            fig.add_trace(go.Scatter3d(
                x=emotion_data['x'],
                y=emotion_data['y'],
                z=emotion_data['z'],
                mode='markers',
                name=emotion.upper(),
                marker=dict(
                    size=6,
                    color=self.emotion_colors.get(emotion, 'gray'),
                    opacity=0.6
                ),
                hovertemplate=f'<b>{emotion.upper()}</b><extra></extra>'
            ))
            
            # Add centroid
            if show_centroids:
                centroid = emotion_data[['x', 'y', 'z']].mean()
                
                fig.add_trace(go.Scatter3d(
                    x=[centroid['x']],
                    y=[centroid['y']],
                    z=[centroid['z']],
                    mode='markers',
                    name=f'{emotion.upper()} Centroid',
                    marker=dict(
                        size=15,
                        color=self.emotion_colors.get(emotion, 'gray'),
                        symbol='diamond',
                        line=dict(color='black', width=2)
                    ),
                    hovertemplate=f'<b>{emotion.upper()} CENTROID</b><extra></extra>'
                ))
        
        fig.update_layout(
            title='Emotion Clusters in 3D Feature Space',
            scene=dict(
                xaxis_title='PC1',
                yaxis_title='PC2',
                zaxis_title='PC3',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            height=700
        )
        
        return fig
