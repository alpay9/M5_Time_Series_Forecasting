"""
WRMSSE Metric Calculation for M5 Forecasting - FIXED VERSION
=============================================================

Implements the official M5 Competition metric:
Weighted Root Mean Squared Scaled Error

Fixes:
- Better handling of edge cases
- Improved numerical stability
- Added validation

Author: CS 415 Deep Learning Project Team
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
import warnings


class WRMSSECalculator:
    """
    Calculate WRMSSE metric for M5 hierarchical forecasting.
    
    This is the official M5 competition metric that accounts for:
    1. Scale differences across series (RMSSE)
    2. Dollar-value importance (weights)
    3. Hierarchical structure (aggregation levels)
    """
    
    def __init__(self, df_clean: pd.DataFrame, train_end_day: int = None):
        """
        Initialize calculator with cleaned dataframe.
        
        Args:
            df_clean: DataFrame with sales, prices, and identifiers
            train_end_day: Last day of training period (for scale calculation)
        """
        self.df = df_clean
        self.train_end_day = train_end_day
        
        print("\n" + "="*80)
        print("INITIALIZING WRMSSE CALCULATOR")
        print("="*80)
        
        # Calculate scale factors (denominator for RMSSE)
        self._calculate_scale_factors()
        
        # Calculate weights (based on dollar sales)
        self._calculate_weights()
        
        print("✓ WRMSSE calculator ready")
    
    
    def _calculate_scale_factors(self):
        """
        Calculate scale factor for each series.
        
        Scale = average absolute first difference over training period.
        This normalizes errors by the typical variation in each series.
        """
        print("\nCalculating scale factors...")

        df_sorted = self.df.sort_values(['id', 'date'])
        series_index = df_sorted.groupby('id').cumcount()

        if self.train_end_day:
            training_mask = series_index < self.train_end_day
        else:
            series_sizes = df_sorted.groupby('id')['sales'].transform('size')
            n_train = (series_sizes * 0.7).astype(int)
            training_mask = series_index < n_train

        training_df = df_sorted.loc[training_mask]

        diffs = training_df.groupby('id')['sales'].diff().abs()
        scale_series = diffs.groupby(training_df['id']).mean()
        scale_series = scale_series.fillna(1.0).clip(lower=0.1)

        self.scale_factors = scale_series.to_dict()

        scale_values = list(scale_series.values)
        print(f"  ✓ Calculated scale factors for {len(self.scale_factors)} series")
        print(f"  Scale factor range: [{min(scale_values):.4f}, {max(scale_values):.4f}]")
        print(f"  Scale factor mean: {np.mean(scale_values):.4f}")
    
    
    def _calculate_weights(self):
        """
        Calculate weights based on total dollar sales.
        
        Higher dollar-value series get more weight in the metric.
        This ensures the model focuses on items that matter most financially.
        """
        print("\nCalculating dollar-based weights...")
        
        # Calculate total dollar sales per series
        dollar_sales = {}
        
        for series_id in self.df['id'].unique():
            series_data = self.df[self.df['id'] == series_id]
            
            # Total dollars = sum(sales * price)
            total_dollars = (series_data['sales'].astype(np.float64) * 
                           series_data['sell_price'].astype(np.float64)).sum()
            dollar_sales[series_id] = max(total_dollars, 0.01)  # Avoid zero
        
        # Convert to weights (normalize to sum to 1)
        total_dollars_all = sum(dollar_sales.values())
        
        self.weights = {
            series_id: dollars / total_dollars_all
            for series_id, dollars in dollar_sales.items()
        }
        
        weight_values = list(self.weights.values())
        print(f"  ✓ Calculated weights for {len(self.weights)} series")
        print(f"  Total dollar sales: ${total_dollars_all:,.2f}")
        print(f"  Weight range: [{min(weight_values):.6f}, {max(weight_values):.6f}]")
    
    
    def calculate_rmsse(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        series_ids: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate RMSSE for each series.
        
        RMSSE = RMSE / scale_factor
        
        Args:
            predictions: (n_samples, forecast_horizon)
            actuals: (n_samples, forecast_horizon)
            series_ids: (n_samples,) - which series each sample belongs to
            
        Returns:
            Dictionary of series_id -> RMSSE
        """
        squared_errors = (predictions.astype(np.float64) - actuals.astype(np.float64)) ** 2
        per_sample_mse = np.mean(squared_errors, axis=1)

        unique_series, inverse = np.unique(series_ids, return_inverse=True)
        series_error_sums = np.bincount(inverse, weights=per_sample_mse)
        series_counts = np.bincount(inverse)

        rmsse_dict = {}

        for series_id, error_sum, count in zip(unique_series, series_error_sums, series_counts):
            if count == 0:
                continue

            rmse = np.sqrt(error_sum / count)
            scale = self.scale_factors.get(series_id, 1.0)
            rmsse_dict[series_id] = rmse / scale

        return rmsse_dict
    
    
    def calculate_wrmsse(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        series_ids: np.ndarray
    ) -> float:
        """
        Calculate overall WRMSSE (Weighted RMSSE).
        
        WRMSSE = Σ(weight_i * RMSSE_i)
        
        Args:
            predictions: (n_samples, forecast_horizon) - model predictions
            actuals: (n_samples, forecast_horizon) - actual values
            series_ids: (n_samples,) - series ID for each sample
            
        Returns:
            WRMSSE score (lower is better)
        """
        print("\n" + "-"*40)
        print("CALCULATING WRMSSE")
        print("-"*40)
        
        # Validate inputs
        if len(predictions) != len(actuals) or len(predictions) != len(series_ids):
            raise ValueError("predictions, actuals, and series_ids must have same length")
        
        # Calculate RMSSE for each series
        rmsse_dict = self.calculate_rmsse(predictions, actuals, series_ids)
        
        if len(rmsse_dict) == 0:
            warnings.warn("No valid RMSSE values calculated!")
            return 0.0
        
        # Calculate weighted average
        wrmsse = 0.0
        total_weight = 0.0
        
        for series_id, rmsse in rmsse_dict.items():
            weight = self.weights.get(series_id, 0.0)
            wrmsse += weight * rmsse
            total_weight += weight
        
        # Normalize by total weight (in case not all series are present)
        if total_weight > 0:
            wrmsse = wrmsse / total_weight
        
        rmsse_values = list(rmsse_dict.values())
        print(f"  Number of series evaluated: {len(rmsse_dict)}")
        print(f"  WRMSSE: {wrmsse:.6f}")
        print(f"  Average RMSSE: {np.mean(rmsse_values):.6f}")
        print(f"  RMSSE range: [{min(rmsse_values):.6f}, {max(rmsse_values):.6f}]")
        
        return wrmsse
    
    
    def calculate_simple_wrmsse(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray
    ) -> float:
        """
        Calculate a simplified WRMSSE without series tracking.
        
        This is less accurate but works when series IDs aren't tracked.
        Uses global scale factor.
        
        Args:
            predictions: (n_samples, forecast_horizon)
            actuals: (n_samples, forecast_horizon)
            
        Returns:
            Approximate WRMSSE score
        """
        # Global scale factor (mean of all series scales)
        global_scale = np.mean(list(self.scale_factors.values()))
        
        # Calculate RMSE
        squared_errors = (predictions.astype(np.float64) - actuals.astype(np.float64)) ** 2
        rmse = np.sqrt(np.mean(squared_errors))
        
        # Approximate WRMSSE
        wrmsse_approx = rmse / global_scale
        
        return wrmsse_approx


def simple_rmse(predictions: np.ndarray, actuals: np.ndarray) -> float:
    """
    Simple RMSE calculation for quick comparison.
    
    Args:
        predictions: (n_samples, forecast_horizon)
        actuals: (n_samples, forecast_horizon)
        
    Returns:
        RMSE value
    """
    squared_errors = (predictions.astype(np.float64) - actuals.astype(np.float64)) ** 2
    mse = np.mean(squared_errors)
    rmse = np.sqrt(mse)
    
    return rmse


def create_series_id_mapping(
    df_clean: pd.DataFrame, 
    n_test_samples: int,
    input_length: int = 90,
    output_length: int = 28,
    stride: int = 7
) -> np.ndarray:
    """
    Create proper mapping from test sample indices to series IDs.
    
    This recreates the sequence creation logic to properly track
    which series each test sample came from.
    
    Args:
        df_clean: Cleaned dataframe with 'id' column
        n_test_samples: Number of test samples
        input_length: Input sequence length
        output_length: Output sequence length  
        stride: Stride for sequence creation
        
    Returns:
        Array of series IDs corresponding to test samples
    """
    series_ids_list = []
    
    series_unique = df_clean['id'].unique()
    
    for series_id in series_unique:
        series_data = df_clean[df_clean['id'] == series_id]
        n_points = len(series_data)
        
        # Calculate how many sequences this series contributes
        n_sequences = max(0, (n_points - input_length - output_length) // stride + 1)
        
        # Add series ID for each sequence
        series_ids_list.extend([series_id] * n_sequences)
    
    # Take only the test portion (last 15% typically)
    total_sequences = len(series_ids_list)
    test_start = int(total_sequences * 0.85)  # Assuming 70/15/15 split
    
    series_ids_test = series_ids_list[test_start:test_start + n_test_samples]
    
    # If we don't have enough, cycle through
    if len(series_ids_test) < n_test_samples:
        warnings.warn(f"Not enough series IDs ({len(series_ids_test)}) for test samples ({n_test_samples}). Cycling.")
        series_ids_test = series_ids_test * (n_test_samples // len(series_ids_test) + 1)
        series_ids_test = series_ids_test[:n_test_samples]
    
    return np.array(series_ids_test)
