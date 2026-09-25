"""
GBM pairwise matcher — LightGBM classifier trained with hard negatives.
Calibrated for macro-F0.5 optimization.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import Dict, List, Tuple, Optional
import os
import pickle
import time

try:
    from ..features.pairwise import FEATURE_COLUMNS, RETRIEVAL_FEATURE_COLUMNS
except ImportError:
    from features.pairwise import FEATURE_COLUMNS, RETRIEVAL_FEATURE_COLUMNS


class PairMatcher:
    """LightGBM binary classifier for pairwise matching."""
    
    def __init__(self, params: Optional[dict] = None):
        self.params = params or {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'learning_rate': 0.05,
            'num_leaves': 63,
            'max_depth': -1,
            'min_child_samples': 50,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'n_estimators': 1000,
            'verbose': -1,
            'n_jobs': -1,
            'is_unbalance': True,
        }
        self.model = None
        self.feature_columns = None
    
    def get_feature_columns(self, df: pd.DataFrame) -> list:
        """Get feature columns present in the dataframe."""
        all_cols = FEATURE_COLUMNS + RETRIEVAL_FEATURE_COLUMNS
        return [c for c in all_cols if c in df.columns]
    
    def train(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None,
        early_stopping_rounds: int = 50,
    ):
        """
        Train the LightGBM matcher.
        
        train_df must have feature columns + 'label' column (1=match, 0=non-match).
        """
        self.feature_columns = self.get_feature_columns(train_df)
        print(f"  [PairMatcher] Training with {len(self.feature_columns)} features, "
              f"{len(train_df):,} samples")
        print(f"    Positive: {(train_df['label']==1).sum():,}, "
              f"Negative: {(train_df['label']==0).sum():,}")
        
        X_train = train_df[self.feature_columns].values
        y_train = train_df['label'].values
        
        callbacks = [lgb.log_evaluation(period=100)]
        
        if val_df is not None:
            X_val = val_df[self.feature_columns].values
            y_val = val_df['label'].values
            
            callbacks.append(lgb.early_stopping(stopping_rounds=early_stopping_rounds))
            
            self.model = lgb.LGBMClassifier(**self.params)
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=callbacks,
            )
        else:
            self.model = lgb.LGBMClassifier(**self.params)
            self.model.fit(X_train, y_train, callbacks=callbacks)
        
        print(f"  [PairMatcher] Training done. Best iteration: {self.model.best_iteration_}")
    
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Predict match probabilities."""
        X = df[self.feature_columns].values
        return self.model.predict_proba(X)[:, 1]
    
    def feature_importance(self) -> pd.DataFrame:
        """Get feature importance."""
        importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_,
        }).sort_values('importance', ascending=False)
        return importance
    
    def save(self, path: str):
        """Save model."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'feature_columns': self.feature_columns,
                'params': self.params,
            }, f)
    
    def load(self, path: str):
        """Load model."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.model = data['model']
        self.feature_columns = data['feature_columns']
        self.params = data['params']
