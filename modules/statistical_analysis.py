"""
Statistical Analysis Module
ANOVA, t-tests, and other statistical significance tests
Publication-ready statistical analysis
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import f_oneway, ttest_ind, chi2_contingency, mannwhitneyu
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import cohen_kappa_score


class StatisticalAnalyzer:
    """
    Perform statistical significance tests for BCI research
    """
    
    def __init__(self, alpha=0.05):
        """
        Args:
            alpha: Significance level (default 0.05)
        """
        self.alpha = alpha
    
    def anova_emotions(self, data_by_emotion):
        """
        Perform one-way ANOVA to test if emotions have significantly different features
        
        Args:
            data_by_emotion: Dict with structure {emotion: [values]}
        
        Returns:
            Dictionary with ANOVA results
        """
        # Extract groups
        groups = [values for values in data_by_emotion.values()]
        
        # Perform ANOVA
        f_stat, p_value = f_oneway(*groups)
        
        # Effect size (eta-squared)
        grand_mean = np.mean(np.concatenate(groups))
        ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in groups)
        ss_total = sum((x - grand_mean)**2 for g in groups for x in g)
        eta_squared = ss_between / ss_total if ss_total > 0 else 0
        
        return {
            'f_statistic': f_stat,
            'p_value': p_value,
            'significant': p_value < self.alpha,
            'eta_squared': eta_squared,
            'interpretation': self._interpret_effect_size(eta_squared),
            'conclusion': f"{'Significant' if p_value < self.alpha else 'Not significant'} difference between emotions (p={p_value:.4f})"
        }
    
    def pairwise_t_tests(self, data_by_emotion, correction='bonferroni'):
        """
        Perform pairwise t-tests between all emotion pairs
        
        Args:
            data_by_emotion: Dict with structure {emotion: [values]}
            correction: Multiple comparison correction ('bonferroni' or 'none')
        
        Returns:
            DataFrame with pairwise comparison results
        """
        emotions = list(data_by_emotion.keys())
        results = []
        
        # Number of comparisons for Bonferroni correction
        n_comparisons = len(emotions) * (len(emotions) - 1) // 2
        adjusted_alpha = self.alpha / n_comparisons if correction == 'bonferroni' else self.alpha
        
        for i, emotion1 in enumerate(emotions):
            for emotion2 in emotions[i+1:]:
                # Perform t-test
                t_stat, p_value = ttest_ind(
                    data_by_emotion[emotion1],
                    data_by_emotion[emotion2]
                )
                
                # Calculate Cohen's d (effect size)
                cohens_d = self._calculate_cohens_d(
                    data_by_emotion[emotion1],
                    data_by_emotion[emotion2]
                )
                
                results.append({
                    'Emotion 1': emotion1.upper(),
                    'Emotion 2': emotion2.upper(),
                    't-statistic': t_stat,
                    'p-value': p_value,
                    'Adjusted p-value': p_value * n_comparisons if correction == 'bonferroni' else p_value,
                    'Significant': p_value < adjusted_alpha,
                    "Cohen's d": cohens_d,
                    'Effect Size': self._interpret_cohens_d(cohens_d)
                })
        
        return pd.DataFrame(results)
    
    def _calculate_cohens_d(self, group1, group2):
        """Calculate Cohen's d effect size"""
        mean1, mean2 = np.mean(group1), np.mean(group2)
        std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
        n1, n2 = len(group1), len(group2)
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
        
        return (mean1 - mean2) / pooled_std if pooled_std > 0 else 0
    
    def _interpret_cohens_d(self, d):
        """Interpret Cohen's d effect size"""
        abs_d = abs(d)
        if abs_d < 0.2:
            return 'Negligible'
        elif abs_d < 0.5:
            return 'Small'
        elif abs_d < 0.8:
            return 'Medium'
        else:
            return 'Large'
    
    def _interpret_effect_size(self, eta_squared):
        """Interpret eta-squared effect size"""
        if eta_squared < 0.01:
            return 'Negligible'
        elif eta_squared < 0.06:
            return 'Small'
        elif eta_squared < 0.14:
            return 'Medium'
        else:
            return 'Large'
    
    def confidence_intervals(self, data, confidence=0.95):
        """
        Calculate confidence intervals
        
        Args:
            data: Array of values
            confidence: Confidence level (default 0.95)
        
        Returns:
            Dictionary with CI information
        """
        mean = np.mean(data)
        sem = stats.sem(data)
        ci = sem * stats.t.ppf((1 + confidence) / 2, len(data) - 1)
        
        return {
            'mean': mean,
            'std': np.std(data, ddof=1),
            'sem': sem,
            'ci_lower': mean - ci,
            'ci_upper': mean + ci,
            'confidence_level': confidence,
            'n': len(data)
        }
    
    def model_comparison_test(self, model1_predictions, model2_predictions, true_labels):
        """
        Statistical comparison of two models using McNemar's test
        
        Args:
            model1_predictions: Predictions from model 1
            model2_predictions: Predictions from model 2
            true_labels: True labels
        
        Returns:
            Dictionary with test results
        """
        # Create contingency table
        model1_correct = (model1_predictions == true_labels)
        model2_correct = (model2_predictions == true_labels)
        
        # McNemar's test
        b = np.sum(model1_correct & ~model2_correct)  # Model 1 correct, Model 2 wrong
        c = np.sum(~model1_correct & model2_correct)  # Model 1 wrong, Model 2 correct
        
        if b + c == 0:
            return {
                'test': "McNemar's Test",
                'statistic': 0,
                'p_value': 1.0,
                'significant': False,
                'conclusion': 'Models perform identically'
            }
        
        # Chi-square statistic with continuity correction
        chi2_stat = (abs(b - c) - 1)**2 / (b + c)
        p_value = 1 - stats.chi2.cdf(chi2_stat, 1)
        
        return {
            'test': "McNemar's Test",
            'statistic': chi2_stat,
            'p_value': p_value,
            'significant': p_value < self.alpha,
            'model1_better': b > c,
            'model2_better': c > b,
            'conclusion': self._interpret_mcnemar(b, c, p_value)
        }
    
    def _interpret_mcnemar(self, b, c, p_value):
        """Interpret McNemar's test results"""
        if p_value >= self.alpha:
            return 'No significant difference between models'
        elif b > c:
            return 'Model 1 performs significantly better'
        else:
            return 'Model 2 performs significantly better'
    
    def inter_rater_agreement(self, rater1, rater2):
        """
        Calculate inter-rater agreement (Cohen's Kappa)
        
        Args:
            rater1: Predictions/ratings from rater 1
            rater2: Predictions/ratings from rater 2
        
        Returns:
            Dictionary with agreement metrics
        """
        kappa = cohen_kappa_score(rater1, rater2)
        
        # Percent agreement
        percent_agreement = np.mean(rater1 == rater2)
        
        return {
            'cohens_kappa': kappa,
            'percent_agreement': percent_agreement,
            'interpretation': self._interpret_kappa(kappa),
            'n': len(rater1)
        }
    
    def _interpret_kappa(self, kappa):
        """Interpret Cohen's Kappa"""
        if kappa < 0:
            return 'Poor (worse than chance)'
        elif kappa < 0.20:
            return 'Slight'
        elif kappa < 0.40:
            return 'Fair'
        elif kappa < 0.60:
            return 'Moderate'
        elif kappa < 0.80:
            return 'Substantial'
        else:
            return 'Almost Perfect'
    
    def plot_anova_results(self, data_by_emotion, title='ANOVA: Emotion Comparison'):
        """
        Visualize ANOVA results with box plots
        
        Args:
            data_by_emotion: Dict with structure {emotion: [values]}
            title: Plot title
        
        Returns:
            plotly figure
        """
        # Prepare data
        plot_data = []
        for emotion, values in data_by_emotion.items():
            for value in values:
                plot_data.append({
                    'Emotion': emotion.upper(),
                    'Value': value
                })
        
        df = pd.DataFrame(plot_data)
        
        # Create box plot
        fig = px.box(
            df,
            x='Emotion',
            y='Value',
            title=title,
            color='Emotion',
            color_discrete_map={
                'POSITIVE': '#2ecc71',
                'NEGATIVE': '#e74c3c',
                'NEUTRAL': '#95a5a6'
            }
        )
        
        # Perform ANOVA
        anova_results = self.anova_emotions(data_by_emotion)
        
        # Add annotation with results
        fig.add_annotation(
            text=f"F={anova_results['f_statistic']:.2f}, p={anova_results['p_value']:.4f}<br>" +
                 f"η²={anova_results['eta_squared']:.3f} ({anova_results['interpretation']})",
            xref="paper", yref="paper",
            x=0.02, y=0.98,
            showarrow=False,
            bgcolor="white",
            bordercolor="black",
            borderwidth=1
        )
        
        fig.update_layout(
            height=500,
            showlegend=False
        )
        
        return fig
    
    def plot_confidence_intervals(self, data_by_group, confidence=0.95):
        """
        Plot confidence intervals for multiple groups
        
        Args:
            data_by_group: Dict with structure {group: [values]}
            confidence: Confidence level
        
        Returns:
            plotly figure
        """
        groups = []
        means = []
        ci_lowers = []
        ci_uppers = []
        
        for group, values in data_by_group.items():
            ci_results = self.confidence_intervals(values, confidence)
            
            groups.append(group.upper())
            means.append(ci_results['mean'])
            ci_lowers.append(ci_results['ci_lower'])
            ci_uppers.append(ci_results['ci_upper'])
        
        # Create error bars
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=groups,
            y=means,
            error_y=dict(
                type='data',
                symmetric=False,
                array=[u - m for u, m in zip(ci_uppers, means)],
                arrayminus=[m - l for m, l in zip(means, ci_lowers)]
            ),
            mode='markers',
            marker=dict(size=12, color='#3498db'),
            name='Mean with CI'
        ))
        
        fig.update_layout(
            title=f'{int(confidence*100)}% Confidence Intervals',
            xaxis_title='Group',
            yaxis_title='Value',
            height=500,
            showlegend=False
        )
        
        return fig


class ABTestingFramework:
    """
    A/B Testing framework for model comparison
    """
    
    def __init__(self, alpha=0.05):
        self.alpha = alpha
        self.analyzer = StatisticalAnalyzer(alpha)
    
    def compare_models(self, model_a_results, model_b_results, true_labels):
        """
        Comprehensive A/B test between two models
        
        Args:
            model_a_results: Predictions from model A
            model_b_results: Predictions from model B
            true_labels: True labels
        
        Returns:
            Dictionary with comprehensive comparison
        """
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        
        # Calculate metrics for both models
        acc_a = accuracy_score(true_labels, model_a_results)
        acc_b = accuracy_score(true_labels, model_b_results)
        
        prec_a, rec_a, f1_a, _ = precision_recall_fscore_support(
            true_labels, model_a_results, average='macro'
        )
        prec_b, rec_b, f1_b, _ = precision_recall_fscore_support(
            true_labels, model_b_results, average='macro'
        )
        
        # Statistical test
        mcnemar_results = self.analyzer.model_comparison_test(
            model_a_results, model_b_results, true_labels
        )
        
        # Effect size
        diff = acc_a - acc_b
        
        return {
            'model_a': {
                'accuracy': acc_a,
                'precision': prec_a,
                'recall': rec_a,
                'f1': f1_a
            },
            'model_b': {
                'accuracy': acc_b,
                'precision': prec_b,
                'recall': rec_b,
                'f1': f1_b
            },
            'difference': {
                'accuracy': diff,
                'precision': prec_a - prec_b,
                'recall': rec_a - rec_b,
                'f1': f1_a - f1_b
            },
            'statistical_test': mcnemar_results,
            'winner': 'Model A' if acc_a > acc_b else 'Model B' if acc_b > acc_a else 'Tie',
            'significant_difference': mcnemar_results['significant']
        }
    
    def plot_ab_comparison(self, comparison_results, model_a_name='Model A', 
                          model_b_name='Model B'):
        """
        Visualize A/B test results
        
        Args:
            comparison_results: Results from compare_models
            model_a_name: Name for model A
            model_b_name: Name for model B
        
        Returns:
            plotly figure
        """
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        
        model_a_scores = [comparison_results['model_a'][m] for m in metrics]
        model_b_scores = [comparison_results['model_b'][m] for m in metrics]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name=model_a_name,
            x=[m.upper() for m in metrics],
            y=model_a_scores,
            marker_color='#3498db'
        ))
        
        fig.add_trace(go.Bar(
            name=model_b_name,
            x=[m.upper() for m in metrics],
            y=model_b_scores,
            marker_color='#e74c3c'
        ))
        
        # Add significance annotation
        sig_text = "✓ Significant" if comparison_results['significant_difference'] else "✗ Not Significant"
        p_val = comparison_results['statistical_test']['p_value']
        
        fig.add_annotation(
            text=f"McNemar's Test: {sig_text}<br>p = {p_val:.4f}",
            xref="paper", yref="paper",
            x=0.02, y=0.98,
            showarrow=False,
            bgcolor="white",
            bordercolor="black",
            borderwidth=1
        )
        
        fig.update_layout(
            title=f'A/B Test: {model_a_name} vs {model_b_name}',
            yaxis_title='Score',
            barmode='group',
            height=500,
            yaxis=dict(range=[0, 1])
        )
        
        return fig
