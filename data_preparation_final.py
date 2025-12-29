"""
M5 Forecasting - Final Data Preparation
========================================

Features:
- RobustScaler for normalization
- Hierarchical aggregation features
- NaN handling via row removal
- Memory-optimized data types
- Full dataset support

Author: CS 415 Deep Learning Project Team
"""

import pandas as pd
import numpy as np
from typing import Tuple, List
from sklearn.preprocessing import RobustScaler
import pickle
import gc


class M5DataPreprocessor:
    """
    Handles M5 dataset preprocessing with all optimizations.
    """
    
    def __init__(self, sales_path: str, calendar_path: str, prices_path: str):
        self.sales_path = sales_path
        self.calendar_path = calendar_path
        self.prices_path = prices_path
        self.scaler = RobustScaler()
        
    def load_data(self, n_series: int = None) -> pd.DataFrame:
        """
        Load and merge M5 data with memory optimization.
        
        Args:
            n_series: Number of series to load (None for all 30,490 series)
            
        Returns:
            Merged dataframe
        """
        print("="*80)
        print("LOADING DATA")
        print("="*80)
        
        # ============ SALES DATA ============
        id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
        dtype_ids = {col: "category" for col in id_cols}
        
        print(f"\nLoading sales data...")
        if n_series:
            print(f"  Loading {n_series} series (subset)")
        else:
            print(f"  Loading ALL series (30,490 total)")
        
        sales = pd.read_csv(
            self.sales_path,
            dtype=dtype_ids,
            nrows=n_series
        )
        
        print(f"✓ Loaded {len(sales)} series")
        
        # Melt to long format
        print("  Melting to long format...")
        sales_long = sales.melt(
            id_vars=id_cols,
            var_name="d",
            value_name="sales"
        )
        
        # Optimize data types
        sales_long["d"] = sales_long["d"].astype("category")
        sales_long["sales"] = sales_long["sales"].astype("int16")
        
        del sales
        gc.collect()
        
        # ============ CALENDAR DATA ============
        print("\nLoading calendar data...")
        calendar = pd.read_csv(self.calendar_path)
        calendar["date"] = pd.to_datetime(calendar["date"])
        
        # Optimize calendar dtypes
        calendar["weekday"] = calendar["weekday"].astype("category")
        calendar["month"] = calendar["month"].astype("int8")
        calendar["year"] = calendar["year"].astype("int16")
        calendar["wday"] = calendar["wday"].astype("int8")
        calendar["event_name_1"] = calendar["event_name_1"].astype("category")
        calendar["event_type_1"] = calendar["event_type_1"].astype("category")
        
        for col in ["snap_CA", "snap_TX", "snap_WI"]:
            calendar[col] = calendar[col].astype("int8")
        
        print("✓ Calendar loaded and optimized")
        
        # ============ PRICE DATA ============
        print("\nLoading price data...")
        sell_prices = pd.read_csv(
            self.prices_path,
            dtype={
                "store_id": "category",
                "item_id": "category",
                "sell_price": "float32"
            }
        )
        print("✓ Prices loaded")
        
        # ============ MERGE ============
        print("\nMerging datasets...")
        
        sales_long = sales_long.merge(calendar, how="left", on="d")
        sales_long = sales_long.merge(
            sell_prices,
            how="left",
            on=["store_id", "item_id", "wm_yr_wk"]
        )
        
        # Fill missing prices
        sales_long["sell_price"] = sales_long.groupby("id", observed=True)["sell_price"].fillna(method='ffill')
        sales_long["sell_price"] = sales_long["sell_price"].fillna(0).astype("float32")
        
        # Sort by id and date (critical for lag features)
        print("\nSorting data...")
        sales_long = sales_long.sort_values(["id", "date"]).reset_index(drop=True)

        # Release large intermediate merge inputs
        del calendar, sell_prices
        gc.collect()
        
        print(f"\n✓ Merge complete")
        print(f"  Final shape: {sales_long.shape}")
        print(f"  Memory usage: {sales_long.memory_usage(deep=True).sum() / 1e9:.2f} GB")
        
        return sales_long
    
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all time series features including hierarchical aggregations.
        """
        print("\n" + "="*80)
        print("FEATURE ENGINEERING")
        print("="*80)
        
        # ============ LAG FEATURES ============
        print("\n1. Creating lag features...")
        for lag in [1, 7, 28]:
            print(f"   - lag_{lag}")
            df[f"lag_{lag}"] = df.groupby("id", observed=True)["sales"].shift(lag).astype("float32")
        
        # ============ ROLLING WINDOW FEATURES ============
        print("\n2. Creating rolling window features...")
        for window in [7, 28]:
            print(f"   - rolling_mean_{window}, rolling_std_{window}")
            df[f"rolling_mean_{window}"] = (
                df.groupby("id", observed=True)["lag_1"]
                .transform(lambda x: x.rolling(window, min_periods=1).mean())
                .astype("float32")
            )
            
            df[f"rolling_std_{window}"] = (
                df.groupby("id", observed=True)["lag_1"]
                .transform(lambda x: x.rolling(window, min_periods=1).std())
                .astype("float32")
            )
        
        # ============ PRICE FEATURES ============
        print("\n3. Creating price features...")
        
        df["price_change"] = (
            df["sell_price"] - df.groupby("id", observed=True)["sell_price"].shift(7)
        ).astype("float32")
        
        df["price_rolling_mean"] = (
            df.groupby("id", observed=True)["sell_price"]
            .transform(lambda x: x.rolling(28, min_periods=1).mean())
            .astype("float32")
        )
        
        df["price_relative"] = (
            df["sell_price"] / (df["price_rolling_mean"] + 1e-5)
        ).astype("float32")
        
        # ============ CALENDAR FEATURES ============
        print("\n4. Creating calendar features...")
        
        df["day_of_week"] = df["wday"].astype("int8")
        df["month_num"] = df["month"].astype("int8")
        df["is_weekend"] = (df["wday"] >= 6).astype("int8")
        df["has_event"] = (~df["event_name_1"].isna()).astype("int8")
        
        # SNAP active per state
        df["snap_active"] = 0
        df.loc[df["state_id"] == "CA", "snap_active"] = df.loc[df["state_id"] == "CA", "snap_CA"]
        df.loc[df["state_id"] == "TX", "snap_active"] = df.loc[df["state_id"] == "TX", "snap_TX"]
        df.loc[df["state_id"] == "WI", "snap_active"] = df.loc[df["state_id"] == "WI", "snap_WI"]
        df["snap_active"] = df["snap_active"].astype("int8")
        
        # ============ ADVANCED FEATURES ============
        print("\n5. Creating advanced features...")
        
        # Cumulative sales
        print("   - cumsum_7")
        df["cumsum_7"] = (
            df.groupby("id", observed=True)["sales"]
            .transform(lambda x: x.rolling(7, min_periods=1).sum())
            .astype("float32")
        )
        
        # Sales velocity
        print("   - sales_velocity")
        df["sales_velocity"] = (
            df["lag_1"] - df["lag_7"]
        ).astype("float32")
        
        # ============ HIERARCHICAL AGGREGATION FEATURES ============
        print("\n6. Creating hierarchical aggregation features...")
        
        # Store total sales
        print("   - store_total_sales")
        store_totals = df.groupby(["store_id", "date"], observed=True)["sales"].sum().rename("store_total_sales")
        df = df.merge(store_totals, on=["store_id", "date"], how="left")
        df["store_total_sales"] = df["store_total_sales"].astype("float32")
        
        # Category total sales
        print("   - category_total_sales")
        category_totals = df.groupby(["cat_id", "date"], observed=True)["sales"].sum().rename("category_total_sales")
        df = df.merge(category_totals, on=["cat_id", "date"], how="left")
        df["category_total_sales"] = df["category_total_sales"].astype("float32")

        del store_totals, category_totals
        gc.collect()
        
        # Item percentage of store
        print("   - item_pct_of_store")
        df["item_pct_of_store"] = (
            df["sales"] / (df["store_total_sales"] + 1)
        ).astype("float32")
        
        # ============ DROP UNNECESSARY COLUMNS ============
        print("\n7. Dropping unnecessary columns...")
        drop_cols = [
            "d", "wm_yr_wk", "weekday", "wday",
            "event_name_1", "event_type_1", "event_name_2", "event_type_2",
            "snap_CA", "snap_TX", "snap_WI"
        ]
        df = df.drop(columns=[col for col in drop_cols if col in df.columns])
        
        print(f"\n✓ Feature engineering complete")
        print(f"  Shape: {df.shape}")
        print(f"  Features created: {df.shape[1] - 8}")  # Minus metadata columns
        print(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1e9:.2f} GB")

        gc.collect()
        
        return df
    
    
    def create_sequences(
        self, 
        df: pd.DataFrame, 
        input_length: int = 90,
        output_length: int = 28,
        stride: int = 7,
        series_batch_size: int = 500,
        use_memmap: bool = True,
        memmap_dir: str = "."
    ) -> Tuple[np.ndarray, np.ndarray, List[str], pd.DataFrame, List[str]]:
        """
        Create sliding window sequences with NaN removal.
        
        Returns:
            X, Y arrays, feature column names, cleaned dataframe for WRMSSE, and series IDs
        """
        print("\n" + "="*80)
        print("SEQUENCE CREATION")
        print("="*80)
        
        # Define feature columns
        exclude_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", 
                       "state_id", "date", "sales", "year"]
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        print(f"\nFeatures to use: {len(feature_cols)}")
        print(f"  {feature_cols}")
        
        # ============ REMOVE NaN ROWS ============
        print(f"\n1. Removing rows with NaN values...")
        original_row_count = len(df)
        print(f"   Before: {original_row_count:,} rows")
        
        df_clean = df.dropna(subset=feature_cols)
        del df
        gc.collect()
        
        removed_rows = original_row_count - len(df_clean)
        removed_pct = (removed_rows / original_row_count * 100) if original_row_count else 0.0
        print(f"   After:  {len(df_clean):,} rows")
        print(f"   Removed: {removed_rows:,} rows ({removed_pct:.2f}%)")
        
        # ============ CREATE SEQUENCES ============
        print(f"\n2. Creating sequences with stride={stride}...")
        print(f"   Input length: {input_length} days")
        print(f"   Output length: {output_length} days")
        
        series_ids = df_clean["id"].unique()
        print(f"   Processing {len(series_ids):,} series...")

        series_lengths = df_clean.groupby("id", observed=True).size()
        total_sequences = 0
        for length in series_lengths:
            max_start = length - input_length - output_length
            if max_start >= 0:
                total_sequences += (max_start // stride) + 1

        print(f"   Total sequences to allocate: {total_sequences:,}")

        series_id_list = np.empty(total_sequences, dtype=object)
        if use_memmap:
            X_path = f"{memmap_dir}/X_sequences.dat"
            Y_path = f"{memmap_dir}/Y_sequences.dat"
            X = np.memmap(
                X_path,
                dtype=np.float32,
                mode="w+",
                shape=(total_sequences, input_length, len(feature_cols))
            )
            Y = np.memmap(
                Y_path,
                dtype=np.float32,
                mode="w+",
                shape=(total_sequences, output_length)
            )
        else:
            X = np.empty((total_sequences, input_length, len(feature_cols)), dtype=np.float32)
            Y = np.empty((total_sequences, output_length), dtype=np.float32)

        write_idx = 0
        for batch_start in range(0, len(series_ids), series_batch_size):
            batch_ids = series_ids[batch_start:batch_start + series_batch_size]
            for idx, series_id in enumerate(batch_ids, start=batch_start):
                if (idx + 1) % 1000 == 0:
                    print(f"   Progress: {idx + 1:,}/{len(series_ids):,} series")

                series_data = df_clean[df_clean["id"] == series_id].sort_values("date")
                features = series_data[feature_cols].values
                targets = np.log1p(series_data["sales"].values)

                for i in range(0, len(series_data) - input_length - output_length + 1, stride):
                    X[write_idx] = features[i:i + input_length]
                    Y[write_idx] = targets[i + input_length:i + input_length + output_length]
                    series_id_list[write_idx] = series_id
                    write_idx += 1

            gc.collect()

        if use_memmap:
            X.flush()
            Y.flush()

        if write_idx != total_sequences:
            raise ValueError(
                f"Sequence count mismatch: expected {total_sequences}, wrote {write_idx}"
            )
        
        print(f"\n✓ Sequence creation complete")
        print(f"  X shape: {X.shape} (samples, timesteps, features)")
        print(f"  Y shape: {Y.shape} (samples, forecast_horizon)")
        print(f"  Total sequences: {len(X):,}")
        
        return X, Y, feature_cols, df_clean, series_id_list
    
    
    def normalize_features(
        self, 
        X_train: np.ndarray, 
        X_val: np.ndarray, 
        X_test: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Normalize features using RobustScaler.
        Fit on train, transform all sets.
        """
        print("\n" + "="*80)
        print("FEATURE NORMALIZATION (RobustScaler)")
        print("="*80)
        
        # Reshape for scaling: (samples * timesteps, features)
        n_samples_train, n_timesteps, n_features = X_train.shape
        n_samples_val = X_val.shape[0]
        n_samples_test = X_test.shape[0]
        
        print(f"\nOriginal shapes:")
        print(f"  Train: {X_train.shape}")
        print(f"  Val:   {X_val.shape}")
        print(f"  Test:  {X_test.shape}")
        
        # Reshape
        X_train_reshaped = X_train.reshape(-1, n_features)
        X_val_reshaped = X_val.reshape(-1, n_features)
        X_test_reshaped = X_test.reshape(-1, n_features)
        
        print(f"\nFitting scaler on training data...")
        self.scaler.fit(X_train_reshaped)
        
        print(f"Transforming all datasets...")
        X_train_scaled = self.scaler.transform(X_train_reshaped)
        X_val_scaled = self.scaler.transform(X_val_reshaped)
        X_test_scaled = self.scaler.transform(X_test_reshaped)
        
        # Reshape back
        X_train_scaled = X_train_scaled.reshape(n_samples_train, n_timesteps, n_features)
        X_val_scaled = X_val_scaled.reshape(n_samples_val, n_timesteps, n_features)
        X_test_scaled = X_test_scaled.reshape(n_samples_test, n_timesteps, n_features)
        
        print(f"\n✓ Normalization complete")
        print(f"  Method: RobustScaler (median=0, IQR=1)")
        print(f"  Scaled shapes:")
        print(f"    Train: {X_train_scaled.shape}")
        print(f"    Val:   {X_val_scaled.shape}")
        print(f"    Test:  {X_test_scaled.shape}")
        
        return X_train_scaled, X_val_scaled, X_test_scaled
    
    
    def split_data(
        self, 
        X: np.ndarray, 
        Y: np.ndarray,
        series_ids: List[str],
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data temporally per series (no shuffle for time series).
        """
        print("\n" + "="*80)
        print("DATA SPLITTING")
        print("="*80)
        
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
        
        series_id_array = np.array(series_ids)
        unique_series_ids = pd.unique(series_id_array)
        print(f"  Splitting {len(unique_series_ids):,} series by time...")
        
        train_indices = []
        val_indices = []
        test_indices = []

        boundaries = np.flatnonzero(series_id_array[1:] != series_id_array[:-1]) + 1
        group_starts = np.r_[0, boundaries]
        group_ends = np.r_[boundaries, len(series_id_array)]
        is_grouped = len(unique_series_ids) == len(group_starts)

        if is_grouped:
            for start, end in zip(group_starts, group_ends):
                n_series_samples = end - start
                train_size = int(n_series_samples * train_ratio)
                val_size = int(n_series_samples * val_ratio)

                train_indices.append(np.arange(start, start + train_size))
                val_indices.append(
                    np.arange(start + train_size, start + train_size + val_size)
                )
                test_indices.append(
                    np.arange(start + train_size + val_size, end)
                )
        else:
            grouped_indices = pd.Series(series_id_array).groupby(series_id_array).indices
            for series_indices in grouped_indices.values():
                series_indices = np.asarray(series_indices)
                n_series_samples = len(series_indices)
                train_size = int(n_series_samples * train_ratio)
                val_size = int(n_series_samples * val_ratio)

                train_indices.append(series_indices[:train_size])
                val_indices.append(series_indices[train_size:train_size + val_size])
                test_indices.append(series_indices[train_size + val_size:])

        train_indices = np.concatenate(train_indices) if train_indices else np.array([], dtype=int)
        val_indices = np.concatenate(val_indices) if val_indices else np.array([], dtype=int)
        test_indices = np.concatenate(test_indices) if test_indices else np.array([], dtype=int)
        
        X_train = X[train_indices]
        Y_train = Y[train_indices]
        
        X_val = X[val_indices]
        Y_val = Y[val_indices]
        
        X_test = X[test_indices]
        Y_test = Y[test_indices]
        
        print(f"\nSplit sizes:")
        print(f"  Train: {len(X_train):,} samples ({train_ratio*100:.0f}%)")
        print(f"  Val:   {len(X_val):,} samples ({val_ratio*100:.0f}%)")
        print(f"  Test:  {len(X_test):,} samples ({test_ratio*100:.0f}%)")
        
        return X_train, Y_train, X_val, Y_val, X_test, Y_test
    
    
    def save_scaler(self, filepath: str):
        """Save the fitted scaler."""
        with open(filepath, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"✓ Scaler saved to {filepath}")


def main():
    """
    Main preprocessing pipeline.
    """
    print("\n" + "="*80)
    print("M5 FORECASTING - DATA PREPARATION PIPELINE")
    print("="*80)
    
    # Initialize
    preprocessor = M5DataPreprocessor(
        sales_path="data/sales_train_evaluation.csv",
        calendar_path="data/calendar.csv",
        prices_path="data/sell_prices.csv"
    )
    
    # Load data (None = all series)
    df = preprocessor.load_data(n_series=None)
    
    # Create features
    df = preprocessor.create_features(df)
    
    # Create sequences
    X, Y, feature_cols, df_clean, series_ids = preprocessor.create_sequences(
        df,
        input_length=90,
        output_length=28,
        stride=7
    )
    
    # Split data
    X_train, Y_train, X_val, Y_val, X_test, Y_test = preprocessor.split_data(
        X,
        Y,
        series_ids
    )
    
    # Normalize
    X_train, X_val, X_test = preprocessor.normalize_features(X_train, X_val, X_test)
    
    # Save everything
    print("\n" + "="*80)
    print("SAVING PROCESSED DATA")
    print("="*80)
    
    np.save("X_train.npy", X_train)
    np.save("Y_train.npy", Y_train)
    np.save("X_val.npy", X_val)
    np.save("Y_val.npy", Y_val)
    np.save("X_test.npy", X_test)
    np.save("Y_test.npy", Y_test)
    print("✓ Saved train/val/test arrays")
    
    with open("feature_cols.txt", "w") as f:
        f.write("\n".join(feature_cols))
    print("✓ Saved feature names")
    
    preprocessor.save_scaler("scaler.pkl")
    
    # Save cleaned dataframe for WRMSSE calculation
    df_clean.to_pickle("df_clean_for_wrmsse.pkl")
    print("✓ Saved cleaned dataframe for WRMSSE")
    
    print("\n" + "="*80)
    print("PREPROCESSING COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print("  - X_train.npy, Y_train.npy")
    print("  - X_val.npy, Y_val.npy")
    print("  - X_test.npy, Y_test.npy")
    print("  - feature_cols.txt")
    print("  - scaler.pkl")
    print("  - df_clean_for_wrmsse.pkl")


if __name__ == "__main__":
    main()
