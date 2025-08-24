"""
Delay prediction models using XGBoost and scikit-learn
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import joblib
import logging
from datetime import datetime
from pathlib import Path

# Scikit-learn imports
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline

# XGBoost import
import xgboost as xgb

# Local imports
from .feature_engineering import FeatureEngineer
from .model_evaluator import ModelEvaluator

logger = logging.getLogger(__name__)


class DelayPredictor:
    """Delay prediction using open source AI tools (XGBoost and scikit-learn)"""
    
    def __init__(self, model_type: str = 'xgboost'):
        """
        Initialize delay predictor
        
        Args:
            model_type: Type of model to use ('xgboost', 'random_forest', 'gradient_boosting', 'linear')
        """
        self.model_type = model_type
        self.model = None
        self.feature_engineer = FeatureEngineer()
        self.evaluator = ModelEvaluator()
        self.is_trained = False
        self.feature_importance = None
        self.model_metrics = {}
        
        # Initialize model based on type
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the specified model type"""
        if self.model_type == 'xgboost':
            self.model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42
            )
        elif self.model_type == 'linear':
            self.model = Ridge(alpha=1.0)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        logger.info(f"Initialized {self.model_type} model")
    
    def prepare_data(self, df: pd.DataFrame, target_column: str = 'delay_minutes') -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for training using feature engineering"""
        logger.info("Preparing data for delay prediction")
        
        # Create features
        X, y = self.feature_engineer.prepare_features(df, target_column)
        
        # Transform features
        X_transformed = self.feature_engineer.fit_transform(X, y)
        
        # Handle missing target values
        if y is not None:
            valid_indices = ~y.isna()
            X_transformed = X_transformed[valid_indices]
            y = y[valid_indices].values
        
        logger.info(f"Data preparation complete. Shape: {X_transformed.shape}")
        return X_transformed, y
    
    def train(self, df: pd.DataFrame, target_column: str = 'delay_minutes', 
              test_size: float = 0.2, cv_folds: int = 5) -> Dict[str, Any]:
        """
        Train the delay prediction model with cross-validation
        
        Args:
            df: Training data
            target_column: Target column name
            test_size: Test set size for train/test split
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dictionary containing training metrics
        """
        logger.info(f"Training {self.model_type} delay prediction model")
        
        # Prepare data
        X, y = self.prepare_data(df, target_column)
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=None
        )
        
        # Cross-validation
        cv_scores = cross_val_score(
            self.model, X_train, y_train, 
            cv=cv_folds, scoring='neg_mean_squared_error', n_jobs=-1
        )
        
        # Train final model
        self.model.fit(X_train, y_train)
        
        # Make predictions
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)
        
        # Calculate metrics
        train_metrics = self.evaluator.calculate_regression_metrics(y_train, y_train_pred)
        test_metrics = self.evaluator.calculate_regression_metrics(y_test, y_test_pred)
        
        # Store metrics
        self.model_metrics = {
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'cv_scores': {
                'mean': -cv_scores.mean(),
                'std': cv_scores.std(),
                'scores': -cv_scores
            },
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'features_count': X.shape[1]
        }
        
        # Feature importance
        self._calculate_feature_importance()
        
        self.is_trained = True
        
        logger.info(f"Model training complete. Test RMSE: {test_metrics['rmse']:.2f}")
        return self.model_metrics
    
    def _calculate_feature_importance(self):
        """Calculate and store feature importance"""
        feature_names = self.feature_engineer.get_feature_names()
        
        if hasattr(self.model, 'feature_importances_'):
            # Tree-based models
            importance_scores = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            # Linear models
            importance_scores = np.abs(self.model.coef_)
        else:
            logger.warning("Model does not support feature importance")
            return
        
        # Create feature importance dataframe
        self.feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_scores
        }).sort_values('importance', ascending=False)
        
        logger.info(f"Top 5 important features: {self.feature_importance.head()['feature'].tolist()}")
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Make delay predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Prepare features
        X, _ = self.feature_engineer.prepare_features(df)
        X_transformed = self.feature_engineer.transform(X)
        
        # Make predictions
        predictions = self.model.predict(X_transformed)
        
        return predictions
    
    def predict_with_confidence(self, df: pd.DataFrame, n_estimators: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with confidence intervals (for tree-based models)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        predictions = self.predict(df)
        
        # For tree-based models, use prediction intervals
        if hasattr(self.model, 'estimators_'):
            # Get predictions from individual trees
            X, _ = self.feature_engineer.prepare_features(df)
            X_transformed = self.feature_engineer.transform(X)
            
            if self.model_type == 'random_forest':
                tree_predictions = np.array([
                    tree.predict(X_transformed) for tree in self.model.estimators_
                ])
                confidence = np.std(tree_predictions, axis=0)
            else:
                # For other models, use a simple heuristic
                confidence = np.abs(predictions) * 0.1  # 10% of prediction as confidence
        else:
            # For linear models, use a simple heuristic
            confidence = np.abs(predictions) * 0.1
        
        return predictions, confidence
    
    def hyperparameter_tuning(self, df: pd.DataFrame, target_column: str = 'delay_minutes') -> Dict[str, Any]:
        """Perform hyperparameter tuning using GridSearchCV"""
        logger.info(f"Starting hyperparameter tuning for {self.model_type}")
        
        # Prepare data
        X, y = self.prepare_data(df, target_column)
        
        # Define parameter grids for different models
        param_grids = {
            'xgboost': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
            },
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'gradient_boosting': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
            },
            'linear': {
                'alpha': [0.1, 1.0, 10.0, 100.0]
            }
        }
        
        param_grid = param_grids.get(self.model_type, {})
        
        if not param_grid:
            logger.warning(f"No parameter grid defined for {self.model_type}")
            return {}
        
        # Perform grid search
        grid_search = GridSearchCV(
            self.model, param_grid, 
            cv=5, scoring='neg_mean_squared_error',
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X, y)
        
        # Update model with best parameters
        self.model = grid_search.best_estimator_
        
        tuning_results = {
            'best_params': grid_search.best_params_,
            'best_score': -grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }
        
        logger.info(f"Hyperparameter tuning complete. Best RMSE: {-grid_search.best_score_:.2f}")
        logger.info(f"Best parameters: {grid_search.best_params_}")
        
        return tuning_results
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Get top N most important features"""
        if self.feature_importance is None:
            raise ValueError("Model must be trained to get feature importance")
        
        return self.feature_importance.head(top_n)
    
    def save_model(self, filepath: str):
        """Save trained model to file"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'model': self.model,
            'feature_engineer': self.feature_engineer,
            'model_type': self.model_type,
            'model_metrics': self.model_metrics,
            'feature_importance': self.feature_importance,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model from file"""
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.feature_engineer = model_data['feature_engineer']
        self.model_type = model_data['model_type']
        self.model_metrics = model_data['model_metrics']
        self.feature_importance = model_data['feature_importance']
        self.is_trained = model_data['is_trained']
        
        logger.info(f"Model loaded from {filepath}")
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get comprehensive model summary"""
        if not self.is_trained:
            return {"status": "Model not trained"}
        
        summary = {
            "model_type": self.model_type,
            "training_status": "trained",
            "metrics": self.model_metrics,
            "top_features": self.get_feature_importance(10).to_dict('records') if self.feature_importance is not None else [],
            "model_parameters": self.model.get_params() if hasattr(self.model, 'get_params') else {}
        }
        
        return summary


class EnsembleDelayPredictor:
    """Ensemble of multiple delay prediction models"""
    
    def __init__(self, model_types: List[str] = ['xgboost', 'random_forest', 'gradient_boosting']):
        """Initialize ensemble with multiple model types"""
        self.model_types = model_types
        self.models = {}
        self.weights = {}
        self.is_trained = False
        
        # Initialize individual models
        for model_type in model_types:
            self.models[model_type] = DelayPredictor(model_type)
    
    def train(self, df: pd.DataFrame, target_column: str = 'delay_minutes') -> Dict[str, Any]:
        """Train all models in the ensemble"""
        logger.info("Training ensemble delay prediction models")
        
        ensemble_metrics = {}
        
        for model_type, model in self.models.items():
            logger.info(f"Training {model_type} model")
            metrics = model.train(df, target_column)
            ensemble_metrics[model_type] = metrics
            
            # Calculate weight based on test performance (inverse of RMSE)
            test_rmse = metrics['test_metrics']['rmse']
            self.weights[model_type] = 1.0 / (test_rmse + 1e-6)
        
        # Normalize weights
        total_weight = sum(self.weights.values())
        self.weights = {k: v / total_weight for k, v in self.weights.items()}
        
        self.is_trained = True
        
        logger.info(f"Ensemble training complete. Model weights: {self.weights}")
        return ensemble_metrics
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Make ensemble predictions using weighted average"""
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before making predictions")
        
        predictions = {}
        for model_type, model in self.models.items():
            predictions[model_type] = model.predict(df)
        
        # Weighted average
        ensemble_pred = np.zeros_like(predictions[self.model_types[0]])
        for model_type, pred in predictions.items():
            ensemble_pred += self.weights[model_type] * pred
        
        return ensemble_pred
    
    def get_individual_predictions(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Get predictions from individual models"""
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before making predictions")
        
        predictions = {}
        for model_type, model in self.models.items():
            predictions[model_type] = model.predict(df)
        
        return predictions