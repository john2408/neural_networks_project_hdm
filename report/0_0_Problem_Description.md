# Automotive Industry Trends Forecasting using Neural Networks

## Introduction 

The automotive industry is undergoing a transformative shift driven by the global transition toward electric mobility and sustainability initiatives. Understanding vehicle registration trends across different powertrain technologies—Electric (BEV), Hybrid, Diesel, and Petrol—is crucial for manufacturers, policymakers, and market analysts to make informed strategic decisions.

This project investigates time series forecasting of vehicle registration patterns in the German automotive market using state-of-the-art neural network architectures. Leveraging monthly registration data from the Kraftfahrt-Bundesamt (KBA) spanning 2018–2025, we analyze registration patterns across multiple Original Equipment Manufacturers (OEMs), vehicle models, and powertrain types at a granular level. The objective is to develop robust forecasting models capable of predicting future registration volumes, thereby providing actionable insights into the evolving landscape of automotive mobility.

The German market serves as an ideal case study given its position as Europe's largest automotive market and its ambitious electrification targets, making accurate forecasting essential for supply chain planning, production scheduling, and market strategy development.

## Problem Description

The challenge lies in forecasting monthly vehicle registration volumes for **1,502 individual time series**, where each series represents a unique combination of OEM, vehicle model, and powertrain type, with at least 12 historical data points and active values up to October 2025. The dataset encompasses:

- **Temporal Coverage:** January 2018 – October 2025 (monthly observations)
- **Total Data Points:** 107,922 observations
- **Average Series Length:** ~62 months per time series
- **Granularity:** Model-level registrations per powertrain type (e.g., BMW X1 Electric, Mercedes A-Class Diesel)

### Key Challenges

1. **High Dimensionality:** Managing 1,502 distinct time series with varying characteristics
2. **Multiple Powertrain Types:** Capturing diverging trends (e.g., rising EV adoption vs. declining diesel registrations)
3. **Market Dynamics:** Accounting for seasonal patterns, policy changes, and economic factors
4. **Data Sparsity:** Some model-powertrain combinations have limited historical data
5. **Structural Breaks:** COVID-19 pandemic effects, semiconductor shortages, and regulatory changes

### Forecasting Objective

### Forecasting Objective

Generate accurate multi-horizon forecasts across three distinct test periods to ensure robust model validation and avoid overfitting:

- **Test Period 1:** October – December 2024
- **Test Period 2:** January – March 2025
- **Test Period 3:** August – October 2025

Each model is trained exclusively on data preceding its respective test period, enabling evaluation across varying market conditions. Performance metrics are averaged across all three folds to establish true model performance.

**Use Cases:**
- **Production Planning:** Optimizing manufacturing schedules and inventory management
- **Market Analysis:** Identifying growth opportunities and declining segments
- **Strategic Decision-Making:** Informing investment in powertrain technologies
- **Policy Evaluation:** Assessing the impact of incentive programs on EV adoption

## Solution Approach

### Data Architecture: Medallion Framework

The project follows the **Medallion Architecture** to ensure data quality and traceability across the machine learning pipeline:

#### 🥉 Bronze Layer (`data/raw/`)
- **Purpose:** Raw, unprocessed data as ingested from source systems
- **Content:** 
  - Original Excel files from KBA FZ10 monthly reports
  - External economic indicators (GDP, interest rates, fuel prices)
  - Sociodemographic data from Stastical Bundesamt
- **Characteristics:** Immutable historical record, no transformations applied

#### 🥈 Silver Layer (`data/processed/`)
- **Purpose:** Cleaned, validated, and standardized data
- **Transformations:**
  - Column name standardization
  - Missing value imputation
  - Data type conversions
  - Wide-to-long format transformation
  - Creation of unique time series identifiers (`ts_key`)
  - Filtering (minimum 12 months of history)
  - Keep timeseries which are active up to Oct 2025
- **Output:** Parquet files optimized for analytics

#### 🥇 Gold Layer (`data/gold/`)
- **Purpose:** Feature-engineered, analysis-ready datasets
- **Enhancements:**
  - Temporal features (year, month, quarter, seasonality indicators)
  - Lag features (historical values at various time steps)
  - Rolling statistics (moving averages, volatility measures)
  - Integration of exogenous variables (economic indicators aligned to time series)
  - Train/validation/test splits with embargo periods to prevent data leakage
- **Output:** Optimized datasets for model training and evaluation

### Forecasting Algorithms

The study benchmarks five state-of-the-art neural network architectures for time series forecasting:

| Algorithm | Type | Key Characteristics |
|-----------|------|---------------------|
| **LSTM** | Recurrent Neural Network | Long Short-Term Memory networks with gating mechanisms to capture long-term dependencies |
| **RNN** | Recurrent Neural Network | Vanilla recurrent architecture serving as baseline for sequential modeling |
| **N-BEATS** | Deep Learning | Neural Basis Expansion Analysis for Time Series with interpretable decomposition |
| **N-BEATSx** | Deep Learning | Extended N-BEATS variant incorporating exogenous variables (economic indicators) |
| **Chronos2** | Foundation Model | Pretrained transformer-based model leveraging zero-shot forecasting capabilities |

### Evaluation Framework

**Primary Metric:** **SMAPE (Symmetric Mean Absolute Percentage Error)**
- **Forecast Horizon:** 3 months (Aug, Sep, Oct 2025)
- **Evaluation Strategy:** Walk-forward validation with embargo periods to prevent lookahead bias
- **Success Criteria:** Minimizing SMAPE across all 1,502 time series

**Model Comparison Criteria:**
- Forecasting accuracy (SMAPE, MAE, RMSE)
- Computational efficiency (training time, inference speed)
- Robustness to outliers and structural breaks
- Ability to leverage exogenous variables
- Scalability to high-dimensional problems

### Implementation Workflow

1. **Data Acquisition:** Automated download and versioning of source data
2. **Data Preprocessing:** Medallion architecture pipeline (Bronze → Silver → Gold)
3. **Exploratory Analysis:** Statistical profiling and visualization of registration trends
4. **Feature Engineering:** Creation of temporal and economic features
5. **Model Development:** Training and hyperparameter tuning for each algorithm
6. **Model Evaluation:** Comparative analysis using SMAPE on holdout period
---

## Project Structure

This report contains the following sections:

```{tableofcontents}
```
