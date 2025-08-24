"""
Model evaluation utilities for flight delay prediction
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error, explained_variance_score
)
from sklearn.model_selection import learning_curve, validation_curve
import logging

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Model evaluation and performance monitoring"""
    
    def __init__(self):
        self.evaluation_history = []
    
    def calculate_regression_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive regression metrics"""
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'mape': mean_absolute_percentage_error(y_true, y_pred) * 100,  # Convert to percentage
            'explained_variance': explained_variance_score(y_true, y_pred),
            'mean_residual': np.mean(y_pred - y_true),
            'std_residual': np.std(y_pred - y_true),
            'max_error': np.max(np.abs(y_pred - y_true))
        }
        
        # Additional custom metrics for delay prediction
        metrics.update(self._calculate_delay_specific_metrics(y_true, y_pred))
        
        return metrics
    
    def _calculate_delay_specific_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate delay-specific evaluation metrics"""
        # Accuracy for delay categories
        delay_threshold = 15  # minutes
        
        # True positives, false positives, etc. for delayed flights
        true_delayed = y_true >= delay_threshold
        pred_delayed = y_pred >= delay_threshold
        
        tp = np.sum(true_delayed & pred_delayed)
        fp = np.sum(~true_delayed & pred_delayed)
        tn = np.sum(~true_delayed & ~pred_delayed)
        fn = np.sum(true_delayed & ~pred_delayed)
        
        # Calculate precision, recall, F1 for delay prediction
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / len(y_true)
        
        # Delay magnitude accuracy (for flights that are actually delayed)
        delayed_mask = y_true >= delay_threshold
        if np.sum(delayed_mask) > 0:
            delayed_mae = mean_absolute_error(y_true[delayed_mask], y_pred[delayed_mask])
            delayed_rmse = np.sqrt(mean_squared_error(y_true[delayed_mask], y_pred[delayed_mask]))
        else:
            delayed_mae = 0
            delayed_rmse = 0
        
        return {
            'delay_precision': precision,
            'delay_recall': recall,
            'delay_f1': f1,
            'delay_accuracy': accuracy,
            'delayed_flights_mae': delayed_mae,
            'delayed_flights_rmse': delayed_rmse,
            'on_time_rate_actual': np.mean(y_true < delay_threshold),
            'on_time_rate_predicted': np.mean(y_pred < delay_threshold)
        }
    
    def evaluate_model_performance(self, model, X_test: np.ndarray, y_test: np.ndarray, 
                                 model_name: str = "Model") -> Dict[str, Any]:
        """Comprehensive model performance evaluation"""
        logger.info(f"Evaluating {model_name} performance")
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = self.calculate_regression_metrics(y_test, y_pred)
        
        # Store evaluation
        evaluation = {
            'model_name': model_name,
            'timestamp': pd.Timestamp.now(),
            'metrics': metrics,
            'sample_size': len(y_test),
            'predictions': y_pred,
            'actuals': y_test
        }
        
        self.evaluation_history.append(evaluation)
        
        return evaluation
    
    def compare_models(self, evaluations: List[Dict[str, Any]]) -> pd.DataFrame:
        """Compare multiple model evaluations"""
        comparison_data = []
        
        for eval_data in evaluations:
            row = {
                'model_name': eval_data['model_name'],
                'sample_size': eval_data['sample_size']
            }
            row.update(eval_data['metrics'])
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Sort by RMSE (lower is better)
        comparison_df = comparison_df.sort_values('rmse')
        
        return comparison_df
    
    def plot_prediction_vs_actual(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                 title: str = "Predictions vs Actual", 
                                 save_path: Optional[str] = None) -> plt.Figure:
        """Plot predictions vs actual values"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Scatter plot
        ax1.scatter(y_true, y_pred, alpha=0.6)
        ax1.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        ax1.set_xlabel('Actual Delay (minutes)')
        ax1.set_ylabel('Predicted Delay (minutes)')
        ax1.set_title(f'{title} - Scatter Plot')
        ax1.grid(True, alpha=0.3)
        
        # Add R² score to the plot
        r2 = r2_score(y_true, y_pred)
        ax1.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax1.transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Residual plot
        residuals = y_pred - y_true
        ax2.scatter(y_pred, residuals, alpha=0.6)
        ax2.axhline(y=0, color='r', linestyle='--')
        ax2.set_xlabel('Predicted Delay (minutes)')
        ax2.set_ylabel('Residuals (minutes)')
        ax2.set_title(f'{title} - Residual Plot')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_feature_importance(self, feature_importance: pd.DataFrame, 
                               top_n: int = 20, 
                               title: str = "Feature Importance",
                               save_path: Optional[str] = None) -> plt.Figure:
        """Plot feature importance"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Get top N features
        top_features = feature_importance.head(top_n)
        
        # Create horizontal bar plot
        bars = ax.barh(range(len(top_features)), top_features['importance'])
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'])
        ax.set_xlabel('Importance Score')
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis='x')
        
        # Color bars by importance
        colors = plt.cm.viridis(np.linspace(0, 1, len(top_features)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_learning_curve(self, model, X: np.ndarray, y: np.ndarray, 
                           title: str = "Learning Curve",
                           cv: int = 5, save_path: Optional[str] = None) -> plt.Figure:
        """Plot learning curve to analyze model performance vs training size"""
        train_sizes, train_scores, val_scores = learning_curve(
            model, X, y, cv=cv, n_jobs=-1, 
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring='neg_mean_squared_error'
        )
        
        # Convert to positive RMSE
        train_rmse = np.sqrt(-train_scores)
        val_rmse = np.sqrt(-val_scores)
        
        train_rmse_mean = np.mean(train_rmse, axis=1)
        train_rmse_std = np.std(train_rmse, axis=1)
        val_rmse_mean = np.mean(val_rmse, axis=1)
        val_rmse_std = np.std(val_rmse, axis=1)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(train_sizes, train_rmse_mean, 'o-', color='blue', label='Training RMSE')
        ax.fill_between(train_sizes, train_rmse_mean - train_rmse_std,
                       train_rmse_mean + train_rmse_std, alpha=0.1, color='blue')
        
        ax.plot(train_sizes, val_rmse_mean, 'o-', color='red', label='Validation RMSE')
        ax.fill_between(train_sizes, val_rmse_mean - val_rmse_std,
                       val_rmse_mean + val_rmse_std, alpha=0.1, color='red')
        
        ax.set_xlabel('Training Set Size')
        ax.set_ylabel('RMSE')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_delay_distribution(self, y_true: np.ndarray, y_pred: np.ndarray,
                               title: str = "Delay Distribution Comparison",
                               save_path: Optional[str] = None) -> plt.Figure:
        """Plot distribution comparison between actual and predicted delays"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Histogram comparison
        bins = np.linspace(min(y_true.min(), y_pred.min()), 
                          max(y_true.max(), y_pred.max()), 50)
        
        ax1.hist(y_true, bins=bins, alpha=0.7, label='Actual', color='blue')
        ax1.hist(y_pred, bins=bins, alpha=0.7, label='Predicted', color='red')
        ax1.set_xlabel('Delay (minutes)')
        ax1.set_ylabel('Frequency')
        ax1.set_title(f'{title} - Histogram')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Box plot comparison
        data_to_plot = [y_true, y_pred]
        ax2.boxplot(data_to_plot, labels=['Actual', 'Predicted'])
        ax2.set_ylabel('Delay (minutes)')
        ax2.set_title(f'{title} - Box Plot')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def generate_model_report(self, evaluation: Dict[str, Any], 
                             feature_importance: Optional[pd.DataFrame] = None) -> str:
        """Generate a comprehensive model evaluation report"""
        metrics = evaluation['metrics']
        model_name = evaluation['model_name']
        
        report = f"""
# Model Evaluation Report: {model_name}

## Model Performance Summary
- **Sample Size**: {evaluation['sample_size']:,}
- **Evaluation Date**: {evaluation['timestamp']}

## Regression Metrics
- **RMSE**: {metrics['rmse']:.2f} minutes
- **MAE**: {metrics['mae']:.2f} minutes
- **R² Score**: {metrics['r2']:.3f}
- **MAPE**: {metrics['mape']:.1f}%
- **Explained Variance**: {metrics['explained_variance']:.3f}

## Delay-Specific Metrics
- **Delay Prediction Accuracy**: {metrics['delay_accuracy']:.1%}
- **Delay Precision**: {metrics['delay_precision']:.3f}
- **Delay Recall**: {metrics['delay_recall']:.3f}
- **Delay F1-Score**: {metrics['delay_f1']:.3f}

## Residual Analysis
- **Mean Residual**: {metrics['mean_residual']:.2f} minutes
- **Residual Std**: {metrics['std_residual']:.2f} minutes
- **Max Error**: {metrics['max_error']:.2f} minutes

## On-Time Performance
- **Actual On-Time Rate**: {metrics['on_time_rate_actual']:.1%}
- **Predicted On-Time Rate**: {metrics['on_time_rate_predicted']:.1%}

## Delayed Flights Analysis
- **Delayed Flights MAE**: {metrics['delayed_flights_mae']:.2f} minutes
- **Delayed Flights RMSE**: {metrics['delayed_flights_rmse']:.2f} minutes
"""
        
        if feature_importance is not None:
            report += "\n## Top 10 Most Important Features\n"
            for i, row in feature_importance.head(10).iterrows():
                report += f"- **{row['feature']}**: {row['importance']:.4f}\n"
        
        return report
    
    def monitor_model_drift(self, baseline_metrics: Dict[str, float], 
                           current_metrics: Dict[str, float],
                           threshold: float = 0.1) -> Dict[str, Any]:
        """Monitor model performance drift"""
        drift_analysis = {
            'drift_detected': False,
            'degraded_metrics': [],
            'metric_changes': {}
        }
        
        for metric, baseline_value in baseline_metrics.items():
            if metric in current_metrics:
                current_value = current_metrics[metric]
                
                # Calculate relative change
                if baseline_value != 0:
                    relative_change = (current_value - baseline_value) / abs(baseline_value)
                else:
                    relative_change = 0
                
                drift_analysis['metric_changes'][metric] = {
                    'baseline': baseline_value,
                    'current': current_value,
                    'relative_change': relative_change,
                    'absolute_change': current_value - baseline_value
                }
                
                # Check for significant degradation (for metrics where lower is better)
                if metric in ['rmse', 'mae', 'mape'] and relative_change > threshold:
                    drift_analysis['drift_detected'] = True
                    drift_analysis['degraded_metrics'].append(metric)
                # Check for significant degradation (for metrics where higher is better)
                elif metric in ['r2', 'delay_accuracy', 'delay_f1'] and relative_change < -threshold:
                    drift_analysis['drift_detected'] = True
                    drift_analysis['degraded_metrics'].append(metric)
        
        return drift_analysis
    
    def get_evaluation_summary(self) -> pd.DataFrame:
        """Get summary of all model evaluations"""
        if not self.evaluation_history:
            return pd.DataFrame()
        
        summary_data = []
        for eval_data in self.evaluation_history:
            row = {
                'model_name': eval_data['model_name'],
                'timestamp': eval_data['timestamp'],
                'sample_size': eval_data['sample_size']
            }
            row.update(eval_data['metrics'])
            summary_data.append(row)
        
        return pd.DataFrame(summary_data)