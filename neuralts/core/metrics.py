import numpy as np
import pandas as pd

def smape(y_true, y_pred):
    """Symmetric Mean Absolute Percentage Error"""
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0
    return 100 * np.mean(diff)

def calculate_smape_distribution(predictions_dict, actuals_dict):
    """
    Calculate SMAPE per time series and categorize into quality buckets.
    
    Args:
        predictions_dict: Dict mapping ts_key -> list of predictions
        actuals_dict: Dict mapping ts_key -> list of actual values
        
    Returns:
        DataFrame with ts_key, smape, and category columns
    """
    smape_results = []
    
    for ts_key in predictions_dict.keys():
        preds = np.array(predictions_dict[ts_key])
        acts = np.array(actuals_dict[ts_key])
        
        # Calculate SMAPE for this time series
        ts_smape = smape(acts, preds)
        
        # Categorize SMAPE
        if ts_smape < 10:
            category = '<10%'
        elif ts_smape <= 20:
            category = '10-20%'
        elif ts_smape <= 30:
            category = '20-30%'
        elif ts_smape <= 40:
            category = '30-40%'
        else:
            category = '>40%'
        
        smape_results.append({
            'ts_key': ts_key,
            'smape': ts_smape,
            'category': category
        })
    
    return pd.DataFrame(smape_results)