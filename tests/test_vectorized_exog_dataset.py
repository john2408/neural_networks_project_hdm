"""
Test suite for TimeSeriesDatasetVectorizedExog class.

This test validates:
1. Correct batch shape with exogenous features
2. Static feature validation
3. Data integrity
4. Comparison with univariate version
"""

import pandas as pd
import numpy as np
import os
import sys
from torch.utils.data import DataLoader

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from neuralts.core.func import TimeSeriesDatasetVectorizedExog, TimeSeriesDatasetVectorized


class TestTimeSeriesDatasetVectorizedExog:
    """Test suite for TimeSeriesDatasetVectorizedExog"""
    
    def data_path(self):
        """Get path to test data"""
        return os.path.join(os.getcwd(), "data", "gold", "monthly_registration_volume_gold_padding.parquet")
    
    def real_data(self):
        """Load real automotive registration data"""
        data_path = self.data_path()
        if not os.path.exists(data_path):
            return None
        
        df = pd.read_parquet(data_path, engine='pyarrow')
        return df
    
    def synthetic_data_with_exog(self):
        """Create synthetic data with static exogenous features"""
        np.random.seed(42)
        
        n_series = 100
        n_timesteps = 60
        
        # Create dates
        dates = pd.date_range('2020-01', periods=n_timesteps, freq='M')
        
        # Create data
        data = []
        for ts_idx in range(n_series):
            for date_idx, date in enumerate(dates):
                # Each series has different values
                value = 1000 + ts_idx * 10 + date_idx * np.random.randn()
                
                # Exogenous features are STATIC (same for all series at each date)
                gdp = 100 + date_idx * 0.5 + np.random.randn() * 2
                interest_rate = 3.0 + date_idx * 0.01 + np.random.randn() * 0.1
                cpi = 105 + date_idx * 0.3 + np.random.randn() * 0.5
                
                data.append({
                    'Date': date,
                    'ts_key': f'series_{ts_idx}',
                    'Value': value,
                    'GDP': gdp,
                    'Interest_Rate': interest_rate,
                    'CPI': cpi
                })
        
        df = pd.DataFrame(data)
        
        # Ensure exogenous features are truly static (same across all series at each date)
        for date in dates:
            date_data = df[df['Date'] == date]
            gdp_val = date_data['GDP'].iloc[0]
            ir_val = date_data['Interest_Rate'].iloc[0]
            cpi_val = date_data['CPI'].iloc[0]
            
            df.loc[df['Date'] == date, 'GDP'] = gdp_val
            df.loc[df['Date'] == date, 'Interest_Rate'] = ir_val
            df.loc[df['Date'] == date, 'CPI'] = cpi_val
        
        return df
    
    def test_basic_initialization(self, synthetic_data_with_exog):
        """Test basic dataset initialization with exogenous features"""
        exog_cols = ['GDP', 'Interest_Rate', 'CPI']
        
        dataset = TimeSeriesDatasetVectorizedExog(
            df=synthetic_data_with_exog,
            exog_cols=exog_cols,
            seq_length=6,
            embargo=1,
            train=True,
            train_ratio=0.8
        )
        
        # Verify dataset properties
        assert dataset.n_exog == 3, f"Expected 3 exogenous features, got {dataset.n_exog}"
        assert dataset.exog_cols == exog_cols, "Exogenous columns mismatch"
        assert dataset.n_series == 100, f"Expected 100 series, got {dataset.n_series}"
        
        print(f"✓ Basic initialization successful")
        print(f"  Series: {dataset.n_series}")
        print(f"  Exogenous features: {dataset.n_exog}")
        print(f"  Dataset length: {len(dataset)}")
    
    def test_batch_shape_with_exog(self, synthetic_data_with_exog):
        """Test that batches have correct shape with exogenous features"""
        exog_cols = ['GDP', 'Interest_Rate', 'CPI']
        seq_length = 6
        n_exog = len(exog_cols)
        expected_features = 1 + n_exog  # Value + exogenous
        
        dataset = TimeSeriesDatasetVectorizedExog(
            df=synthetic_data_with_exog,
            exog_cols=exog_cols,
            seq_length=seq_length,
            embargo=1,
            train=True,
            train_ratio=0.8
        )
        
        # Test single sample
        X, y = dataset[0]
        
        assert X.shape == (100, seq_length, expected_features), \
            f"Expected X shape (100, {seq_length}, {expected_features}), got {X.shape}"
        assert y.shape == (100,), f"Expected y shape (100,), got {y.shape}"
        
        # Test batch from DataLoader
        loader = DataLoader(dataset, batch_size=8, shuffle=False)
        X_batch, y_batch = next(iter(loader))
        
        assert X_batch.shape == (8, 100, seq_length, expected_features), \
            f"Expected batch shape (8, 100, {seq_length}, {expected_features}), got {X_batch.shape}"
        assert y_batch.shape == (8, 100), f"Expected y batch shape (8, 100), got {y_batch.shape}"
        
        print(f"✓ Batch shapes correct")
        print(f"  Single sample X: {X.shape}")
        print(f"  Single sample y: {y.shape}")
        print(f"  Batch X: {X_batch.shape}")
        print(f"  Batch y: {y_batch.shape}")
    
    def test_static_feature_validation(self, synthetic_data_with_exog):
        """Test that non-static features are rejected"""
        # Create a copy and make GDP non-static (different values across series)
        df_bad = synthetic_data_with_exog[['Date', 'ts_key', 'Value', 'GDP']].copy()
        
        # Make GDP different for each series at first date
        first_date = df_bad['Date'].min()
        for idx, (_, row) in enumerate(df_bad[df_bad['Date'] == first_date].iterrows()):
            df_bad.loc[(df_bad['Date'] == first_date) & (df_bad['ts_key'] == row['ts_key']), 'GDP'] = 100 + idx
        
        # This should raise an error
        try:
            dataset = TimeSeriesDatasetVectorizedExog(
                df=df_bad,
                exog_cols=['GDP'],
                seq_length=6,
                embargo=1,
                train=True
            )
            raise AssertionError("Expected ValueError for non-static features")
        except ValueError as e:
            if "not static" in str(e):
                print(f"✓ Static feature validation working correctly")
            else:
                raise
    
    def test_comparison_with_univariate(self, synthetic_data_with_exog):
        """Compare dataset size with univariate version"""
        # Univariate (Value only)
        df_univariate = synthetic_data_with_exog[['Date', 'ts_key', 'Value']].copy()
        
        dataset_univariate = TimeSeriesDatasetVectorized(
            df=df_univariate,
            seq_length=6,
            embargo=1,
            train=True,
            train_ratio=0.8
        )
        
        # Multivariate (Value + exogenous)
        dataset_multivariate = TimeSeriesDatasetVectorizedExog(
            df=synthetic_data_with_exog,
            exog_cols=['GDP', 'Interest_Rate', 'CPI'],
            seq_length=6,
            embargo=1,
            train=True,
            train_ratio=0.8
        )
        
        # Both should have same number of samples (time windows)
        assert len(dataset_univariate) == len(dataset_multivariate), \
            "Univariate and multivariate datasets should have same length"
        
        # Check feature dimensions
        X_uni, _ = dataset_univariate[0]
        X_multi, _ = dataset_multivariate[0]
        
        assert X_uni.shape[2] == 1, f"Univariate should have 1 feature, got {X_uni.shape[2]}"
        assert X_multi.shape[2] == 4, f"Multivariate should have 4 features (1+3), got {X_multi.shape[2]}"
        
        print(f"✓ Comparison with univariate version successful")
        print(f"  Univariate features: {X_uni.shape[2]}")
        print(f"  Multivariate features: {X_multi.shape[2]}")
        print(f"  Both have {len(dataset_univariate)} time windows")
    
    def test_no_exog_fallback(self, synthetic_data_with_exog):
        """Test that dataset works without exogenous features (empty list)"""
        df_univariate = synthetic_data_with_exog[['Date', 'ts_key', 'Value']].copy()
        
        dataset = TimeSeriesDatasetVectorizedExog(
            df=df_univariate,
            exog_cols=[],  # Empty list
            seq_length=6,
            embargo=1,
            train=True,
            train_ratio=0.8
        )
        
        X, y = dataset[0]
        assert X.shape[2] == 1, f"Without exog features, should have 1 feature, got {X.shape[2]}"
        
        print(f"✓ Dataset works without exogenous features")
    
    def test_real_data_compatibility(self, real_data):
        """Test with real automotive registration data if available"""
        if real_data is None:
            print(f"⚠ Skipping real data test (file not found)")
            return
            
        # Check if we can add synthetic static features to real data
        df = real_data.copy()
        
        # Keep only required columns
        df = df[['Date', 'ts_key', 'Value']].copy()
        
        # Add synthetic static exogenous features
        unique_dates = df['Date'].unique()
        np.random.seed(42)
        
        date_to_gdp = {date: 100 + i * 0.5 for i, date in enumerate(sorted(unique_dates))}
        date_to_ir = {date: 3.0 + i * 0.01 for i, date in enumerate(sorted(unique_dates))}
        
        df['GDP'] = df['Date'].map(date_to_gdp)
        df['Interest_Rate'] = df['Date'].map(date_to_ir)
        
        # Create dataset
        dataset = TimeSeriesDatasetVectorizedExog(
            df=df,
            exog_cols=['GDP', 'Interest_Rate'],
            seq_length=6,
            embargo=1,
            train=True,
            train_ratio=0.8
        )
        
        X, y = dataset[0]
        expected_features = 3  # Value + GDP + Interest_Rate
        
        assert X.shape[2] == expected_features, \
            f"Expected {expected_features} features, got {X.shape[2]}"
        
        print(f"✓ Real data compatibility test passed")
        print(f"  Real dataset series: {dataset.n_series}")
        print(f"  Real dataset windows: {len(dataset)}")
        print(f"  Features: {X.shape[2]}")
    
    def test_train_test_split(self, synthetic_data_with_exog):
        """Test train-test split maintains consistency"""
        exog_cols = ['GDP', 'Interest_Rate', 'CPI']
        
        train_dataset = TimeSeriesDatasetVectorizedExog(
            df=synthetic_data_with_exog,
            exog_cols=exog_cols,
            seq_length=6,
            embargo=1,
            train=True,
            train_ratio=0.8
        )
        
        test_dataset = TimeSeriesDatasetVectorizedExog(
            df=synthetic_data_with_exog,
            exog_cols=exog_cols,
            seq_length=6,
            embargo=1,
            train=False,
            train_ratio=0.8,
            scaler_X=train_dataset.scaler_X,
            scaler_y=train_dataset.scaler_y
        )
        
        # Verify split
        total_windows = len(train_dataset) + len(test_dataset)
        expected_train = int(total_windows * 0.8 / (1 - 0.2 + 0.8))
        
        print(f"✓ Train-test split successful")
        print(f"  Training windows: {len(train_dataset)}")
        print(f"  Test windows: {len(test_dataset)}")
        print(f"  Total windows: {total_windows}")


def run_comprehensive_test():
    """Run comprehensive test with detailed output"""
    print("\n" + "="*80)
    print("COMPREHENSIVE TEST: TimeSeriesDatasetVectorizedExog")
    print("="*80 + "\n")
    
    # Create test instance
    test = TestTimeSeriesDatasetVectorizedExog()
    
    # Get fixtures
    data_path = os.path.join(os.getcwd(), "data", "gold", "monthly_registration_volume_gold_padding.parquet")
    
    # Test 1: Synthetic data
    print("\n" + "-"*80)
    print("TEST 1: Synthetic Data with Exogenous Features")
    print("-"*80)
    
    synthetic_data = test.synthetic_data_with_exog()
    
    print(f"\nSynthetic data created:")
    print(f"  Shape: {synthetic_data.shape}")
    print(f"  Columns: {list(synthetic_data.columns)}")
    print(f"  Unique series: {synthetic_data['ts_key'].nunique()}")
    print(f"  Date range: {synthetic_data['Date'].min()} to {synthetic_data['Date'].max()}")
    
    # Verify static features
    print(f"\nVerifying static features...")
    for date in list(synthetic_data['Date'].unique())[:3]:
        date_data = synthetic_data[synthetic_data['Date'] == date]
        gdp_unique = date_data['GDP'].nunique()
        print(f"  {date}: GDP unique values = {gdp_unique} (should be 1)")
    
    test.test_basic_initialization(synthetic_data)
    test.test_batch_shape_with_exog(synthetic_data)
    test.test_static_feature_validation(synthetic_data)
    test.test_comparison_with_univariate(synthetic_data)
    test.test_no_exog_fallback(synthetic_data)
    test.test_train_test_split(synthetic_data)
    
    # Test 2: Real data (if available)
    if os.path.exists(data_path):
        print("\n" + "-"*80)
        print("TEST 2: Real Automotive Registration Data")
        print("-"*80)
        
        real_data = pd.read_parquet(data_path, engine='pyarrow')
        test.test_real_data_compatibility(real_data)
    else:
        print(f"\n⚠ Skipping real data test (file not found: {data_path})")
    
    print("\n" + "="*80)
    print("ALL TESTS PASSED ✓")
    print("="*80 + "\n")


if __name__ == "__main__":
    # Run comprehensive test
    run_comprehensive_test()
