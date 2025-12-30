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
from typing import Dict, Optional, Iterable, List, Tuple, Any
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
        
        self.aggregation_levels = self._get_aggregation_levels()
        self.series_id_to_agg_ids = self._build_series_id_to_agg_ids()

        # Calculate scale factors (denominator for RMSSE)
        self._calculate_scale_factors()
        
        # Calculate weights (based on dollar sales)
        self._calculate_weights()
        
        print("✓ WRMSSE calculator ready")

    def _get_aggregation_levels(self) -> List[Tuple[str, List[str]]]:
        return [
            ("total", []),
            ("state", ["state_id"]),
            ("store", ["store_id"]),
            ("category", ["cat_id"]),
            ("department", ["dept_id"]),
            ("item", ["item_id"]),
            ("state_category", ["state_id", "cat_id"]),
            ("state_department", ["state_id", "dept_id"]),
            ("store_category", ["store_id", "cat_id"]),
            ("store_department", ["store_id", "dept_id"]),
            ("state_item", ["state_id", "item_id"]),
            ("store_item", ["store_id", "item_id"]),
        ]

    def _make_agg_id(self, level_name: str, group_key: Any) -> str:
        if isinstance(group_key, tuple):
            key_str = "_".join(str(value) for value in group_key)
        else:
            key_str = str(group_key)
        return f"{level_name}:{key_str}"

    def _build_series_id_to_agg_ids(self) -> Dict[str, List[str]]:
        base_columns = ["id", "state_id", "store_id", "cat_id", "dept_id", "item_id"]
        base_df = self.df[base_columns].drop_duplicates("id")
        mapping: Dict[str, List[str]] = {}

        for row in base_df.itertuples(index=False):
            row_dict = row._asdict()
            agg_ids: List[str] = []
            for level_name, group_cols in self.aggregation_levels:
                if not group_cols:
                    agg_id = self._make_agg_id(level_name, "all")
                else:
                    group_key = tuple(row_dict[col] for col in group_cols)
                    if len(group_key) == 1:
                        group_key = group_key[0]
                    agg_id = self._make_agg_id(level_name, group_key)
                agg_ids.append(agg_id)
            mapping[row_dict["id"]] = agg_ids

        return mapping

    def _get_training_masks(self) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        df_sorted = self.df.sort_values(["id", "date"])
        series_index = df_sorted.groupby("id").cumcount()

        if self.train_end_day:
            training_mask = series_index < self.train_end_day
            train_cutoff = self.train_end_day
        else:
            series_sizes = df_sorted.groupby("id")["sales"].transform("size")
            train_cutoff = (series_sizes * 0.7).astype(int)
            training_mask = series_index < train_cutoff

        last_28_mask = training_mask & (series_index >= (train_cutoff - 28))

        return df_sorted, training_mask, last_28_mask

    def _calculate_aggregated_scales(self, df: pd.DataFrame) -> pd.Series:
        scale_factors: Dict[str, float] = {}

        for level_name, group_cols in self.aggregation_levels:
            if group_cols:
                grouped = df.groupby(group_cols + ["date"], dropna=False)["sales"].sum().reset_index()
                for group_key, group_df in grouped.groupby(group_cols, dropna=False):
                    series = group_df.sort_values("date")["sales"]
                    diff_sq = series.diff() ** 2
                    scale = diff_sq.mean()
                    agg_id = self._make_agg_id(level_name, group_key)
                    scale_factors[agg_id] = scale
            else:
                total_series = df.groupby("date")["sales"].sum().sort_index()
                diff_sq = total_series.diff() ** 2
                scale_factors[self._make_agg_id(level_name, "all")] = diff_sq.mean()

        return pd.Series(scale_factors)

    def _calculate_aggregated_dollar_sales(self, df: pd.DataFrame) -> Dict[str, float]:
        dollar_sales: Dict[str, float] = {}

        for level_name, group_cols in self.aggregation_levels:
            if group_cols:
                grouped = df.groupby(group_cols, dropna=False)["dollar_sales"].sum()
                for group_key, total in grouped.items():
                    agg_id = self._make_agg_id(level_name, group_key)
                    dollar_sales[agg_id] = max(float(total), 0.01)
            else:
                total = df["dollar_sales"].sum()
                dollar_sales[self._make_agg_id(level_name, "all")] = max(float(total), 0.01)

        return dollar_sales

    def build_aggregated_series(
        self,
        df: Optional[pd.DataFrame] = None
    ) -> Dict[str, pd.Series]:
        """
        Build the official 12 aggregation levels as time series.

        Args:
            df: Optional dataframe to aggregate. Defaults to full dataframe.

        Returns:
            Dictionary of aggregated series keyed by level/group id.
        """
        df_use = self.df if df is None else df
        aggregated_series: Dict[str, pd.Series] = {}

        for level_name, group_cols in self.aggregation_levels:
            if group_cols:
                grouped = df_use.groupby(group_cols + ["date"], dropna=False)["sales"].sum().reset_index()
                for group_key, group_df in grouped.groupby(group_cols, dropna=False):
                    series = group_df.sort_values("date").set_index("date")["sales"]
                    agg_id = self._make_agg_id(level_name, group_key)
                    aggregated_series[agg_id] = series
            else:
                series = df_use.groupby("date")["sales"].sum().sort_index()
                aggregated_series[self._make_agg_id(level_name, "all")] = series

        return aggregated_series

    def _aggregate_predictions_actuals(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        series_ids: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        aggregated_predictions: Dict[str, np.ndarray] = {}
        aggregated_actuals: Dict[str, np.ndarray] = {}

        for pred, act, series_id in zip(predictions, actuals, series_ids):
            agg_ids = self.series_id_to_agg_ids.get(series_id)
            if not agg_ids:
                continue
            for agg_id in agg_ids:
                if agg_id not in aggregated_predictions:
                    aggregated_predictions[agg_id] = pred.astype(np.float64).copy()
                    aggregated_actuals[agg_id] = act.astype(np.float64).copy()
                else:
                    aggregated_predictions[agg_id] += pred
                    aggregated_actuals[agg_id] += act

        return aggregated_predictions, aggregated_actuals
    
    
    def _calculate_scale_factors(self):
        """
        Calculate scale factor for each series.
        
        Scale = mean squared first difference over training period.
        This normalizes errors by the typical variation in each series.
        """
        print("\nCalculating scale factors...")

        df_sorted, training_mask, _ = self._get_training_masks()
        training_df = df_sorted.loc[training_mask]

        scale_series = self._calculate_aggregated_scales(training_df)
        scale_series = scale_series.fillna(1.0).clip(lower=0.1)

        self.scale_factors = scale_series.to_dict()

        scale_values = list(scale_series.values)
        print(f"  ✓ Calculated scale factors for {len(self.scale_factors)} aggregated series")
        print(f"  Scale factor range: [{min(scale_values):.4f}, {max(scale_values):.4f}]")
        print(f"  Scale factor mean: {np.mean(scale_values):.4f}")
    
    
    def _calculate_weights(self):
        """
        Calculate weights based on total dollar sales.
        
        Higher dollar-value series get more weight in the metric.
        This ensures the model focuses on items that matter most financially.
        """
        print("\nCalculating dollar-based weights...")

        df_sorted, _, last_28_mask = self._get_training_masks()
        last_28_df = df_sorted.loc[last_28_mask].copy()
        last_28_df['dollar_sales'] = (
            last_28_df['sales'].astype(np.float64)
            * last_28_df['sell_price'].astype(np.float64)
        )

        dollar_sales = self._calculate_aggregated_dollar_sales(last_28_df)

        # Convert to weights (normalize to sum to 1)
        total_dollars_all = sum(dollar_sales.values())

        if total_dollars_all == 0:
            warnings.warn("Total dollar sales are zero; defaulting to uniform weights.")
            uniform_weight = 1.0 / max(len(dollar_sales), 1)
            self.weights = {series_id: uniform_weight for series_id in dollar_sales.keys()}
        else:
            self.weights = {
                series_id: dollars / total_dollars_all
                for series_id, dollars in dollar_sales.items()
            }
        
        weight_values = list(self.weights.values())
        print(f"  ✓ Calculated weights for {len(self.weights)} aggregated series")
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
            predictions: (n_samples, forecast_horizon) for base series
            actuals: (n_samples, forecast_horizon) for base series
            series_ids: (n_samples,) - base series ID for each sample
            
        Returns:
            Dictionary of aggregated series_id -> RMSSE
        """
        aggregated_predictions, aggregated_actuals = self._aggregate_predictions_actuals(
            predictions, actuals, series_ids
        )

        rmsse_dict = {}
        for series_id, agg_predictions in aggregated_predictions.items():
            agg_actuals = aggregated_actuals[series_id]
            squared_errors = (agg_predictions.astype(np.float64) - agg_actuals.astype(np.float64)) ** 2
            mse = np.mean(squared_errors)
            rmse = np.sqrt(mse)
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
