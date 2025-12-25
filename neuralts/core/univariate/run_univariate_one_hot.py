import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import warnings
import time
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from neuralts.core.models import (LSTMForecaster, RNNForecaster, GRUForecaster, 
                            CNN1DForecaster, MLPForecaster, TransformerForecaster, TransformerForecasterCLS)
from neuralts.core.metrics import smape, calculate_smape_distribution
from neuralts.core.func import TimeSeriesDataset, generate_out_of_sample_predictions, train_epoch, evaluate


warnings.filterwarnings('ignore')


if __name__ == "__main__":


    # ========================================================================
    # TRAINING PARAMETERS
    # ========================================================================

    MODE = "UNI"  # Options: "UNI": Univariate (Only Month and Year), "EXO": Exogenous Variable (All features)
    
    SEQ_LENGTH = 6
    TRAIN_RATIO = 0.8
    EMBARGO = 1
    EPOCHS = 25
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-5

    MLP_LAYERS = 2
    MLP_HIDDEN_SIZE = 512
    MLP_DROPOUT = 0.2

    LSTM_LAYERS = 2
    LSTM_HIDDEN_SIZE = 128
    LSTM_DROPOUT = 0.2

    RNN_LAYERS = 2
    RNN_HIDDEN_SIZE = 128
    RNN_DROPOUT = 0.2

    GRU_LAYERS = 2
    GRU_HIDDEN_SIZE = 128
    GRU_DROPOUT = 0.2

    CNN_LAYERS = 3
    CNN_HIDDEN_SIZE = 64
    CNN_DROPOUT = 0.2

    TRANSFORMER_D_MODEL = 64
    TRANSFORMER_NHEAD = 4
    TRANSFORMER_LAYERS = 2
    TRANSFORMER_DIM_FEEDFORWARD = 256
    TRANSFORMER_DROPOUT = 0.2

    # HYPERPARAMETER OPTIMIZATION WITH OPTUNA
    OPTIMIZE_HYPERPARAMETERS = True  # Set to False to skip optimization
    N_TRIALS = 3  # Number of Optuna trials
    OPTUNA_TIMEOUT = 900  # Timeout in seconds (15 minutes)

    MODEL = 'LSTM'  # Options: 'LSTM', 'RNN', 'GRU', 'CNN1D', 'MLP', 'Transformer', 'BASELINE'

    TOTAL_SCRIPT_RUNTIME = None
    TOTAL_TRAINING_TIME_FOLDS = dict()
    TOTAL_OPTIMIZATION_TIME = None

    # ========================================================================

    SCRIPT_START_TIME = time.time()

    # Check MPS availability
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"✓ MPS device is available")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✓ CUDA device is available")
    else:
        device = torch.device("cpu")
        print(f"Using CPU")
        
    print(f"Device: {device}")



    # Load data
    import os
    cwd = os.getcwd()
    full_path = os.path.join(cwd, "data", "gold", "monthly_registration_volume_gold.parquet")
    output_path = os.path.join(cwd, "models", MODEL.lower() + "_" + MODE.lower() + "_one_hot")
    os.makedirs(output_path, exist_ok=True)

    df_full = pd.read_parquet(full_path, engine='pyarrow')
    df_full['Year'] = df_full['Date'].dt.year
    df_full['Month'] = df_full['Date'].dt.month

    date_col = 'Date'
    ts_key_col = 'ts_key'
    value_col = 'Value'
    #features = [col for col in df_full.columns if col not in [date_col, ts_key_col, value_col]]
    
    if MODE == "UNI":
        features = ['Year', 'Month']
    elif MODE == "MULTI":
        features = [col for col in df_full.columns if col not in [date_col, ts_key_col, value_col]]

    #Validate not NaN or infinite values in features
    assert not df_full[features].isna().any().any(), "NaN values found in features"
    assert not np.isinf(df_full[features].select_dtypes(include=[np.number])).any().any(), "Infinite values found in features"

    print("="*80)
    print("DATA OVERVIEW")
    print("="*80)
    print(f"Original data shape: {df_full.shape}")
    print(f"Unique time series: {df_full.ts_key.nunique()}")
    print(f"Date range: {df_full.Date.min()} to {df_full.Date.max()}")
    print(f"Total observations: {len(df_full):,}")
    

    # Check original data
    print(f"\nOriginal DataFrame:")
    print(f"  NaN values: {df_full.isna().sum().sum()}")
    print(f"  Inf values: {np.isinf(df_full.select_dtypes(include=[np.number])).sum().sum()}")
    print(f"  Value range: [{df_full['Value'].min():.2f}, {df_full['Value'].max():.2f}]")
    
    # Check for zero/very small values that could cause division issues
    print(f"\nZero values in 'Value' column: {(df_full['Value'] == 0).sum()}")
    print(f"Values < 0.01: {(df_full['Value'] < 0.01).sum()}")
    
    # ========================================================================
    # FOLD CONFIGURATION - As per Problem Description
    # ========================================================================
    
    folds = [
        {
            'name': 'Fold 1',
            'train_end': '2024-09-30',    # Train on data up to Sep 2024
            'test_start': '2024-10-01',   # Test on Oct-Dec 2024
            'test_end': '2024-12-31'
        },
        {
            'name': 'Fold 2',
            'train_end': '2024-12-31',    # Train on data up to Dec 2024
            'test_start': '2025-01-01',   # Test on Jan-Mar 2025
            'test_end': '2025-03-31'
        },
        {
            'name': 'Fold 3',
            'train_end': '2025-06-30',    # Train on data up to Jun 2025
            'test_start': '2025-07-01',   # Test on Jul-Sep 2025
            'test_end': '2025-09-30'
        }
    ]
    
    print("\n" + "="*80)
    print("FOLD CONFIGURATION")
    print("="*80)
    for fold in folds:
        print(f"\n{fold['name']}:")
        print(f"  Training data: up to {fold['train_end']}")
        print(f"  Test period: {fold['test_start']} to {fold['test_end']}")
    
    
    # ========================================================================
    # HYPERPARAMETER OPTIMIZATION WITH OPTUNA
    # ========================================================================
    
    best_trial = None
    if OPTIMIZE_HYPERPARAMETERS and MODEL not in ['BASELINE']:
        import optuna
        from neuralts.core.func import create_univariate_model_from_trial, train_with_early_stopping
        
        print("\n" + "="*80)
        print("HYPERPARAMETER OPTIMIZATION WITH OPTUNA")
        print("="*80)
        print(f"Model: {MODEL}")
        print(f"Trials: {N_TRIALS}")
        print(f"Timeout: {OPTUNA_TIMEOUT}s")
        
        # Use Fold 1 data for hyperparameter optimization
        fold_config = folds[0]
        df_train = df_full[df_full['Date'] <= fold_config['train_end']].copy()
        
        print(f"\nUsing {fold_config['name']} for optimization")
        print(f"  Training observations: {len(df_train):,}")
        
        # Create datasets
        train_dataset = TimeSeriesDataset(
            df_train,
            feature_cols=features,
            seq_length=SEQ_LENGTH,
            embargo=EMBARGO,
            train=True,
            train_ratio=TRAIN_RATIO
        )
        
        val_dataset = TimeSeriesDataset(
            df_train,
            feature_cols=features,
            seq_length=SEQ_LENGTH,
            train=False,
            embargo=EMBARGO,
            train_ratio=TRAIN_RATIO,
            scaler_X=train_dataset.scaler_X,
            scaler_y=train_dataset.scaler_y
        )
        
        INPUT_SIZE_OPT = train_dataset.X.shape[2]
        
        print(f"\nOptimization dataset:")
        print(f"  Training samples: {len(train_dataset):,}")
        print(f"  Validation samples: {len(val_dataset):,}")
        print(f"  Input size: {INPUT_SIZE_OPT}")
        
        def objective(trial):
            """
            Optuna objective function for hyperparameter optimization.
            """
            # Suggest hyperparameters
            learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
            batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
            
            # Create model with trial hyperparameters
            model, hyperparams = create_univariate_model_from_trial(
                MODEL, trial, INPUT_SIZE_OPT, SEQ_LENGTH
            )
            model = model.to(device)
            
            # Create data loaders
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            
            # Train with early stopping
            best_val_loss, _ = train_with_early_stopping(
                model, train_loader, val_loader, device,
                learning_rate=learning_rate,
                weight_decay=WEIGHT_DECAY,
                max_epochs=30,
                patience=5
            )
            
            return best_val_loss
        
        # Create and run study
        study = optuna.create_study(
            direction='minimize',
            study_name=f'{MODEL}_optimization',
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        )
        
        print("\nStarting optimization...")
        OPTIMIZATION_START_TIME = time.time()
        study.optimize(objective, n_trials=N_TRIALS, timeout=OPTUNA_TIMEOUT, show_progress_bar=True)
        TOTAL_OPTIMIZATION_TIME = (time.time() - OPTIMIZATION_START_TIME) / 60.0  # in minutes
        print(f"\nOptimization completed in {TOTAL_OPTIMIZATION_TIME:.1f} minutes")
        
        print("\n" + "="*80)
        print("OPTIMIZATION RESULTS")
        print("="*80)
        print(f"Number of finished trials: {len(study.trials)}")
        print(f"\nBest trial:")
        best_trial = study.best_trial
        print(f"  Value (Validation Loss): {best_trial.value:.6f}")
        print(f"  Hyperparameters:")
        for key, value in best_trial.params.items():
            print(f"    {key}: {value}")
        
        # Update hyperparameters with best trial values
        BATCH_SIZE = best_trial.params.get('batch_size', BATCH_SIZE)
        LEARNING_RATE = best_trial.params.get('learning_rate', LEARNING_RATE)
        
        if MODEL == 'MLP':
            MLP_LAYERS = best_trial.params.get('num_layers', MLP_LAYERS)
            MLP_HIDDEN_SIZE = best_trial.params.get('hidden_size', MLP_HIDDEN_SIZE)
            MLP_DROPOUT = best_trial.params.get('dropout', MLP_DROPOUT)
        
        elif MODEL == 'LSTM':
            LSTM_LAYERS = best_trial.params.get('num_layers', LSTM_LAYERS)
            LSTM_HIDDEN_SIZE = best_trial.params.get('hidden_size', LSTM_HIDDEN_SIZE)
            LSTM_DROPOUT = best_trial.params.get('dropout', LSTM_DROPOUT)
        
        elif MODEL == 'RNN':
            RNN_LAYERS = best_trial.params.get('num_layers', RNN_LAYERS)
            RNN_HIDDEN_SIZE = best_trial.params.get('hidden_size', RNN_HIDDEN_SIZE)
            RNN_DROPOUT = best_trial.params.get('dropout', RNN_DROPOUT)
        
        elif MODEL == 'GRU':
            GRU_LAYERS = best_trial.params.get('num_layers', GRU_LAYERS)
            GRU_HIDDEN_SIZE = best_trial.params.get('hidden_size', GRU_HIDDEN_SIZE)
            GRU_DROPOUT = best_trial.params.get('dropout', GRU_DROPOUT)
        
        elif MODEL == 'CNN1D':
            CNN_LAYERS = best_trial.params.get('num_layers', CNN_LAYERS)
            CNN_HIDDEN_SIZE = best_trial.params.get('hidden_size', CNN_HIDDEN_SIZE)
            CNN_DROPOUT = best_trial.params.get('dropout', CNN_DROPOUT)
        
        elif MODEL in ['Transformer', 'TransformerCLS']:
            TRANSFORMER_D_MODEL = best_trial.params.get('d_model', TRANSFORMER_D_MODEL)
            TRANSFORMER_NHEAD = best_trial.params.get('nhead', TRANSFORMER_NHEAD)
            TRANSFORMER_LAYERS = best_trial.params.get('num_layers', TRANSFORMER_LAYERS)
            TRANSFORMER_DIM_FEEDFORWARD = best_trial.params.get('dim_feedforward', TRANSFORMER_DIM_FEEDFORWARD)
            TRANSFORMER_DROPOUT = best_trial.params.get('dropout', TRANSFORMER_DROPOUT)
        
        print(f"\n✓ Hyperparameters updated with best trial values")
        print(f"  Learning rate: {LEARNING_RATE:.6f}")
        print(f"  Batch size: {BATCH_SIZE}")
        
        # Save optimization results
        optuna_results_path = os.path.join(output_path, 'optuna_results.csv')
        trials_df = study.trials_dataframe()
        trials_df.to_csv(optuna_results_path, index=False)
        print(f"\n✓ Saved Optuna results to: {optuna_results_path}")
        
        # Save best hyperparameters
        best_params_path = os.path.join(output_path, 'best_hyperparameters.txt')
        with open(best_params_path, 'w') as f:
            f.write(f"Best Hyperparameters for {MODEL}\n")
            f.write("="*60 + "\n\n")
            f.write(f"Validation Loss: {best_trial.value:.6f}\n\n")
            f.write("Parameters:\n")
            for key, value in best_trial.params.items():
                f.write(f"  {key}: {value}\n")
        print(f"✓ Saved best hyperparameters to: {best_params_path}")
    
    else:
        print("\n⚠ Hyperparameter optimization disabled (OPTIMIZE_HYPERPARAMETERS=False)")
    
    
    # ========================================================================
    # FOLD-WISE TRAINING AND EVALUATION
    # ========================================================================
    
    fold_metrics = []
    fold_smape_distributions = []
    import matplotlib.pyplot as plt
    
    for fold_idx, fold_config in enumerate(folds):
        
        # Fold Variables
        fold_output_dir = os.path.join(output_path, f"fold_{fold_idx + 1}")
        os.makedirs(fold_output_dir, exist_ok=True)

        print("\n" + "="*80)
        print(f"{fold_config['name'].upper()}: {fold_config['test_start']} to {fold_config['test_end']}")
        print("="*80)
        
        # Filter training data up to train_end date
        df_train = df_full[df_full['Date'] <= fold_config['train_end']].copy()
        
        print(f"\nFiltered training data up to {fold_config['train_end']}")
        print(f"  Training observations: {len(df_train):,}")
        print(f"  Date range: {df_train['Date'].min()} to {df_train['Date'].max()}")
        
        # -----------------------------------------------------------------
        # BASELINE MODEL: Moving Average
        # -----------------------------------------------------------------
        if MODEL == 'BASELINE':
            print("\n" + "-"*60)
            print("BASELINE MODEL: Calculating Moving Average Predictions")
            print("-"*60)
            
            # Get test period data
            df_test_period = df_full[
                (df_full['Date'] >= fold_config['test_start']) & 
                (df_full['Date'] <= fold_config['test_end'])
            ].copy()
            
            # Calculate moving average baseline for each time series
            predictions_dict = {}
            actuals_dict = {}
            all_preds = []
            all_acts = []
            
            unique_ts_keys = df_test_period['ts_key'].unique()
            print(f"Generating baseline predictions for {len(unique_ts_keys)} time series...")
            
            for ts_key in unique_ts_keys:
                # Get last SEQ_LENGTH values up to train_end for this time series
                ts_train_data = df_train[df_train['ts_key'] == ts_key].sort_values('Date')
                
                if len(ts_train_data) < SEQ_LENGTH + EMBARGO:
                    # If not enough history, use all available data
                    baseline_value = ts_train_data['Value'].mean()
                else:
                    # Use last SEQ_LENGTH values, skipping the EMBARGO period
                    baseline_value = ts_train_data['Value'].iloc[-(SEQ_LENGTH + EMBARGO):-EMBARGO].mean()

                # Get actual values for this time series in test period
                ts_test_data = df_test_period[df_test_period['ts_key'] == ts_key].sort_values('Date')
                actual_values = ts_test_data['Value'].values
                
                # Predict the same baseline value for all test periods
                predicted_values = np.full(len(actual_values), baseline_value)

                # Round prediction to 1 decimal place
                predicted_values = np.round(predicted_values, 1)
                
                # Replance NaN prediction with zero
                predicted_values = np.where(np.isnan(predicted_values), 0, predicted_values)

                predictions_dict[ts_key] = predicted_values
                actuals_dict[ts_key] = actual_values
                all_preds.extend(predicted_values)
                all_acts.extend(actual_values)
            
            all_preds = np.array(all_preds)
            all_acts = np.array(all_acts)
            
            print(f"✓ Generated {len(all_preds)} baseline predictions")
         
        else:
            # -----------------------------------------------------------------
            # STEP 1: Create datasets for model development (train/val split)
            # -----------------------------------------------------------------
            print("\n" + "-"*60)
            print("STEP 1: Creating datasets for model development")
            print("-"*60)
        
            train_dataset = TimeSeriesDataset(
                df_train,
                feature_cols=features, 
                seq_length=SEQ_LENGTH,
                embargo=EMBARGO,
                train=True,
                train_ratio=TRAIN_RATIO
            )
            
            print(f"Training samples: {len(train_dataset):,}")
            
            # Save scalers and metadata
            scaler_X = train_dataset.scaler_X
            scaler_y = train_dataset.scaler_y
            n_ts_keys = train_dataset.n_ts_keys
            ts_key_to_idx = train_dataset.ts_key_to_idx
            
            test_dataset = TimeSeriesDataset(
                df_train,
                feature_cols=features,
                seq_length=SEQ_LENGTH,
                train=False,
                embargo=EMBARGO,
                train_ratio=TRAIN_RATIO,
                scaler_X=scaler_X,
                scaler_y=scaler_y
            )
            
            print(f"Validation samples: {len(test_dataset):,}")
        
        # -----------------------------------------------------------------
        # STEP 2: Initialize and train model
        # -----------------------------------------------------------------
            print("\n" + "-"*60)
            print(f"STEP 2: Training {MODEL} model")
            print("-"*60)
            
            INPUT_SIZE = train_dataset.X.shape[2]
            
            if MODEL == 'LSTM':
                model = LSTMForecaster(
                    input_size=INPUT_SIZE,
                    hidden_size=LSTM_HIDDEN_SIZE,
                    num_layers=LSTM_LAYERS,
                    dropout=LSTM_DROPOUT
                ).to(device)
            elif MODEL == 'RNN':
                model = RNNForecaster(
                    input_size=INPUT_SIZE,
                    hidden_size=RNN_HIDDEN_SIZE,
                    num_layers=RNN_LAYERS,
                    dropout=RNN_DROPOUT
                ).to(device)
            elif MODEL == 'GRU':
                model = GRUForecaster(
                    input_size=INPUT_SIZE,
                    hidden_size=GRU_HIDDEN_SIZE,
                    num_layers=GRU_LAYERS,
                    dropout=GRU_DROPOUT
                ).to(device)
            elif MODEL == 'CNN1D':
                model = CNN1DForecaster(
                    input_size=INPUT_SIZE,
                    hidden_size=CNN_HIDDEN_SIZE,
                    num_layers=CNN_LAYERS,
                    dropout=CNN_DROPOUT
                ).to(device)
            elif MODEL == 'MLP':
                model = MLPForecaster(
                    input_size=INPUT_SIZE,
                    seq_length=SEQ_LENGTH,
                    hidden_size=MLP_HIDDEN_SIZE,
                    num_layers=MLP_LAYERS,
                    dropout=MLP_DROPOUT
                ).to(device)
            elif MODEL == 'Transformer':
                model = TransformerForecaster(
                    input_size=INPUT_SIZE,
                    d_model=TRANSFORMER_D_MODEL,
                    nhead=TRANSFORMER_NHEAD,
                    num_layers=TRANSFORMER_LAYERS,
                    dim_feedforward=TRANSFORMER_DIM_FEEDFORWARD,
                    dropout=TRANSFORMER_DROPOUT
                ).to(device)
            elif MODEL == 'TransformerCLS':
                model = TransformerForecasterCLS(
                    input_size=INPUT_SIZE,
                    d_model=TRANSFORMER_D_MODEL,
                    nhead=TRANSFORMER_NHEAD,
                    num_layers=TRANSFORMER_LAYERS,
                    dim_feedforward=TRANSFORMER_DIM_FEEDFORWARD,
                    dropout=TRANSFORMER_DROPOUT
                ).to(device)
            else:
                raise ValueError(f"Unsupported MODEL type: {MODEL}")
        
            print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
            
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
            
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
            
            best_val_loss = float('inf')
            train_losses = []
            val_losses = []
            
            start_time = time.time()
            
            for epoch in range(EPOCHS):
                train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
                val_loss = evaluate(model, val_loader, criterion, device)
                
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                
                scheduler.step(val_loss)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = model.state_dict().copy()
                
                if (epoch + 1) % 10 == 0 or epoch == 0:
                    elapsed = time.time() - start_time
                    print(f"Epoch [{epoch+1:3d}/{EPOCHS}] | Train: {train_loss:.6f} | Val: {val_loss:.6f} | Time: {elapsed:.1f}s")
            
            # Load best model
            model.load_state_dict(best_model_state)

            training_time = (time.time() - start_time) / 60.0  # in minutes

            TOTAL_TRAINING_TIME_FOLDS[fold_config['name']] = training_time
            
            print(f"\nTraining completed in {training_time:.1f} minutes")
            print(f"Best validation loss: {best_val_loss:.6f}")
            

            # -----------------------------------------------------------------
            # STEP 3: Out-of-sample predictions on test period
            # -----------------------------------------------------------------

            # Get test period data
            df_test_period = df_full[
                (df_full['Date'] >= fold_config['test_start']) & 
                (df_full['Date'] <= fold_config['test_end'])
            ].copy()


            predictions_dict, actuals_dict, all_preds, all_acts = generate_out_of_sample_predictions(model=model,
                df_test_period=df_test_period,
                df_full=df_full,
                fold_config=fold_config,
                features=features,
                scaler_X=scaler_X,
                scaler_y=scaler_y,
                ts_key_to_idx=ts_key_to_idx,
                n_ts_keys=n_ts_keys,
                seq_length=SEQ_LENGTH,
                embargo=EMBARGO,
                device=device
            )


        # -----------------------------------------------------------------
        # Create predictions DataFrame and save as CSV
        # -----------------------------------------------------------------
        predictions_data = []
        for ts_key, preds in predictions_dict.items():
            acts = actuals_dict[ts_key]
            # Get the dates for this time series in the test period
            ts_group = df_test_period[df_test_period['ts_key'] == ts_key].sort_values('Date')
            dates = ts_group['Date'].values[:len(preds)]
            
            for i, (date, pred, actual) in enumerate(zip(dates, preds, acts)):
                predictions_data.append({
                    'ts_key': ts_key,
                    'Date': pd.to_datetime(date),
                    'pred': pred,
                    'true': actual
                })
        
        predictions_df = pd.DataFrame(predictions_data)
        predictions_csv_path = os.path.join(fold_output_dir, 'predictions.csv')
        predictions_df.to_csv(predictions_csv_path, index=False)
        print(f"✓ Saved predictions to: predictions.csv ({len(predictions_df)} rows)")
        
        # -----------------------------------------------------------------
        # Calculate SMAPE distribution
        # -----------------------------------------------------------------
        smape_distribution_df = calculate_smape_distribution(predictions_dict, actuals_dict)
        
        # Calculate category counts and percentages
        category_counts = smape_distribution_df['category'].value_counts()
        total_series = len(smape_distribution_df)
        
        # Define category order
        category_order = ['<10%', '10-20%', '20-30%', '30-40%', '>40%']
        category_distribution = {}
        
        for cat in category_order:
            count = category_counts.get(cat, 0)
            percentage = (count / total_series * 100) if total_series > 0 else 0
            category_distribution[cat] = {
                'count': count,
                'percentage': percentage
            }
        
        fold_smape_distributions.append({
            'fold': fold_config['name'],
            **{f'{cat}_count': category_distribution[cat]['count'] for cat in category_order},
            **{f'{cat}_pct': category_distribution[cat]['percentage'] for cat in category_order}
        })
        
        # Save SMAPE distribution per time series
        smape_distribution_df.to_csv(os.path.join(fold_output_dir, 'smape_per_ts.csv'), index=False)
        
        print(f"\nSMAPE Distribution for {fold_config['name']}:")
        print(f"  Time series evaluated: {total_series}")
        for cat in category_order:
            count = category_distribution[cat]['count']
            pct = category_distribution[cat]['percentage']
            print(f"  {cat:>10}: {count:4d} series ({pct:5.1f}%)")
        
        # -----------------------------------------------------------------
        # STEP 4: Calculate metrics
        # -----------------------------------------------------------------
        print("\n" + "-"*60)
        print("STEP 4: Evaluation Metrics")
        print("-"*60)
        
        mse = mean_squared_error(all_acts, all_preds)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(all_acts, all_preds)
        r2 = r2_score(all_acts, all_preds)
        smape_score = smape(all_acts, all_preds)
        
        fold_metrics.append({
            'fold': fold_config['name'],
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'smape': smape_score
        })
        
        print(f"\n{fold_config['name']} Out-of-Sample Metrics:")
        print(f"  MSE:   {mse:.2f}")
        print(f"  RMSE:  {rmse:.2f}")
        print(f"  MAE:   {mae:.2f}")
        print(f"  R²:    {r2:.4f}")
        print(f"  SMAPE: {smape_score:.2f}%")
        
        # Save model checkpoint
        if MODEL not in ['BASELINE']:
            torch.save({
                'model_state_dict': best_model_state,
                'input_size': INPUT_SIZE,
                'hidden_size': 64,
                'num_layers': 2,
                'dropout': 0.2,
                'scaler_X': scaler_X,
                'scaler_y': scaler_y,
                'n_ts_keys': n_ts_keys,
                'ts_key_to_idx': ts_key_to_idx,
                'seq_length': SEQ_LENGTH,
                'fold_config': fold_config,
                'metrics': fold_metrics[-1]
            }, os.path.join(fold_output_dir, 'model_checkpoint.pth'))
        
        # Visualization
        if MODEL not in ['BASELINE']:
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            # Training history
            axes[0].plot(train_losses, label='Training Loss', linewidth=2, color='#2E86AB')
            axes[0].plot(val_losses, label='Validation Loss', linewidth=2, color='#A23B72')
            axes[0].set_xlabel('Epoch', fontsize=12)
            axes[0].set_ylabel('Loss (MSE)', fontsize=12)
            axes[0].set_title(f'{fold_config["name"]} - Training History', fontsize=13, fontweight='bold')
            axes[0].legend(fontsize=10)
            axes[0].grid(True, alpha=0.3)
            
            # Predictions vs Actuals
            axes[1].scatter(all_acts, all_preds, alpha=0.3, s=10, color='#2E86AB')
            axes[1].plot([all_acts.min(), all_acts.max()], 
                    [all_acts.min(), all_acts.max()], 
                    'r--', linewidth=2, label='Perfect Prediction')
            axes[1].set_xlabel('Actual Values', fontsize=12)
            axes[1].set_ylabel('Predicted Values', fontsize=12)
            axes[1].set_title(f'{fold_config["name"]} - Predictions (R²={r2:.4f}, SMAPE={smape_score:.2f}%)', 
                    fontsize=13, fontweight='bold')
            axes[1].legend(fontsize=10)
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(fold_output_dir, 'fold_results.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        # Predictions vs Actuals (for all models including BASELINE)
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(all_acts, all_preds, alpha=0.3, s=10, color='#2E86AB')
        ax.plot([all_acts.min(), all_acts.max()], 
            [all_acts.min(), all_acts.max()], 
            'r--', linewidth=2, label='Perfect Prediction')
        ax.set_xlabel('Actual Values', fontsize=12)
        ax.set_ylabel('Predicted Values', fontsize=12)
        ax.set_title(f'{fold_config["name"]} - Predictions (R²={r2:.4f}, SMAPE={smape_score:.2f}%)', 
                 fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(fold_output_dir, 'predictions_vs_actuals.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n✓ Saved fold results to: {fold_output_dir}")

    # ========================================================================
    # SMAPE DISTRIBUTION ANALYSIS
    # ========================================================================
    
    print("\n" + "="*80)
    print("SMAPE DISTRIBUTION ANALYSIS")
    print("="*80)
    
    smape_dist_df = pd.DataFrame(fold_smape_distributions)
    
    print("\nPer-Fold SMAPE Distribution:")
    category_order = ['<10%', '10-20%', '20-30%', '30-40%', '>40%']
    
    for _, row in smape_dist_df.iterrows():
        print(f"\n{row['fold']}:")
        for cat in category_order:
            count = row[f'{cat}_count']
            pct = row[f'{cat}_pct']
            print(f"  {cat:>10}: {count:4.0f} series ({pct:5.1f}%)")
    
    print("\n" + "-"*60)
    print("AVERAGE SMAPE DISTRIBUTION ACROSS FOLDS:")
    print("-"*60)
    
    for cat in category_order:
        avg_count = smape_dist_df[f'{cat}_count'].mean()
        avg_pct = smape_dist_df[f'{cat}_pct'].mean()
        std_pct = smape_dist_df[f'{cat}_pct'].std()
        print(f"  {cat:>10}: {avg_pct:5.1f}% ± {std_pct:4.1f}% ({avg_count:.0f} series avg)")
    
    # Save SMAPE distribution summary
    smape_dist_df.to_csv(os.path.join(output_path, 'smape_distribution.csv'), index=False)
    
    print(f"\n✓ Saved metrics to: {output_path}/fold_metrics.csv")
    print(f"✓ Saved summary to: {output_path}/summary_metrics.csv")
    print(f"✓ Saved SMAPE distribution to: {output_path}/smape_distribution.csv")
    
    # ========================================================================
    # FINAL RESULTS: Average metrics across all folds
    # ========================================================================
    
    print("\n" + "="*80)
    print("FINAL RESULTS: AVERAGE METRICS ACROSS ALL FOLDS")
    print("="*80)
    
    metrics_df = pd.DataFrame(fold_metrics)
    
    print("\nPer-Fold Metrics:")
    print(metrics_df.to_string(index=False))
    
    print("\n" + "-"*60)
    print("AVERAGE PERFORMANCE:")
    print("-"*60)
    avg_metrics = metrics_df[['mse', 'rmse', 'mae', 'r2', 'smape']].mean()
    std_metrics = metrics_df[['mse', 'rmse', 'mae', 'r2', 'smape']].std()
    
    print(f"  MSE:   {avg_metrics['mse']:.2f} ± {std_metrics['mse']:.2f}")
    print(f"  RMSE:  {avg_metrics['rmse']:.2f} ± {std_metrics['rmse']:.2f}")
    print(f"  MAE:   {avg_metrics['mae']:.2f} ± {std_metrics['mae']:.2f}")
    print(f"  R²:    {avg_metrics['r2']:.4f} ± {std_metrics['r2']:.4f}")
    print(f"  SMAPE: {avg_metrics['smape']:.2f}% ± {std_metrics['smape']:.2f}%")
    
    # Save metrics to CSV
    metrics_df.to_csv(os.path.join(output_path, 'fold_metrics.csv'), index=False)
    
    # Save summary
    summary_dict = {
        'metric': ['MSE', 'RMSE', 'MAE', 'R²', 'SMAPE'],
        'mean': [avg_metrics['mse'], avg_metrics['rmse'], avg_metrics['mae'], 
                 avg_metrics['r2'], avg_metrics['smape']],
        'std': [std_metrics['mse'], std_metrics['rmse'], std_metrics['mae'], 
                std_metrics['r2'], std_metrics['smape']]
    }
    summary_df = pd.DataFrame(summary_dict)
    summary_df.to_csv(os.path.join(output_path, 'summary_metrics.csv'), index=False)

    TOTAL_SCRIPT_RUNTIME = (time.time() - SCRIPT_START_TIME) / 60.0  # in minutes
    print(f"\nTotal script runtime: {TOTAL_SCRIPT_RUNTIME:.1f} minutes")
    
    # ========================================================================
    # SAVE COMPREHENSIVE RESULTS TO TXT FILE
    # ========================================================================
    
    results_txt_path = os.path.join(output_path, 'final_results_summary.txt')
    
    with open(results_txt_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"{MODEL} MODEL - FINAL EVALUATION RESULTS\n")
        f.write("="*80 + "\n\n")

        f.write(f"Model: {MODEL}\n")
        if best_trial is not None:
            f.write("Best Hyperparameters:\n")
            for key, value in best_trial.params.items():
                f.write(f"    {key}: {value}\n")
        
        # Total optimization time
        if OPTIMIZE_HYPERPARAMETERS and best_trial is not None:
            f.write(f"\nTotal Hyperparameter Optimization Time: {TOTAL_OPTIMIZATION_TIME:.1f} minutes\n")

        # Training time per fold
        f.write("\nTraining Time per Fold:\n")
        for fold_name, training_time in TOTAL_TRAINING_TIME_FOLDS.items():
            f.write(f"  {fold_name}: {training_time:.1f} minutes\n")
        
        f.write(f"\nTotal Script Runtime: {TOTAL_SCRIPT_RUNTIME:.1f} minutes\n")


        f.write("\n" + "="*80 + "\n\n")
        
        # 1. FINAL RESULTS: AVERAGE METRICS ACROSS ALL FOLDS
        f.write("="*80 + "\n")
        f.write("1. FINAL RESULTS: AVERAGE METRICS ACROSS ALL FOLDS\n")
        f.write("="*80 + "\n\n")
        
        
        f.write("Per-Fold Metrics:\n")
        f.write(metrics_df.to_string(index=False) + "\n\n")
        
        # 4. AVERAGE PERFORMANCE
        f.write("-"*60 + "\n")
        f.write("4. AVERAGE PERFORMANCE:\n")
        f.write("-"*60 + "\n")
        f.write(f"  MSE:   {avg_metrics['mse']:.2f} ± {std_metrics['mse']:.2f}\n")
        f.write(f"  RMSE:  {avg_metrics['rmse']:.2f} ± {std_metrics['rmse']:.2f}\n")
        f.write(f"  MAE:   {avg_metrics['mae']:.2f} ± {std_metrics['mae']:.2f}\n")
        f.write(f"  R²:    {avg_metrics['r2']:.4f} ± {std_metrics['r2']:.4f}\n")
        f.write(f"  SMAPE: {avg_metrics['smape']:.2f}% ± {std_metrics['smape']:.2f}%\n\n")
        
        # 2. SMAPE DISTRIBUTION ANALYSIS per fold
        f.write("="*80 + "\n")
        f.write("2. SMAPE DISTRIBUTION ANALYSIS PER FOLD\n")
        f.write("="*80 + "\n\n")
        
        for _, row in smape_dist_df.iterrows():
            f.write(f"{row['fold']}:\n")
            for cat in category_order:
                count = row[f'{cat}_count']
                pct = row[f'{cat}_pct']
                f.write(f"  {cat:>10}: {count:4.0f} series ({pct:5.1f}%)\n")
            f.write("\n")
        
        # 3. AVERAGE SMAPE DISTRIBUTION ACROSS FOLDS
        f.write("-"*60 + "\n")
        f.write("3. AVERAGE SMAPE DISTRIBUTION ACROSS FOLDS:\n")
        f.write("-"*60 + "\n")
        
        for cat in category_order:
            avg_count = smape_dist_df[f'{cat}_count'].mean()
            avg_pct = smape_dist_df[f'{cat}_pct'].mean()
            std_pct = smape_dist_df[f'{cat}_pct'].std()
            f.write(f"  {cat:>10}: {avg_pct:5.1f}% ± {std_pct:4.1f}% ({avg_count:.0f} series avg)\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*80 + "\n")
    
    print(f"\n✓ Saved metrics to: {output_path}/fold_metrics.csv")
    print(f"✓ Saved summary to: {output_path}/summary_metrics.csv")
    print(f"✓ Saved final results summary to: {output_path}/final_results_summary.txt")
    print("\n" + "="*80)
    
    
    