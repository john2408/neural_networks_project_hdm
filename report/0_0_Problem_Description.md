# Time Series Forecasting using Neural Networks

## Introduction 

The problem at hand refers to forecasting the inbound material volume (in tons) on monthly basis for the next 4 months for an international automotive company. 

The motivation behind that was the lack of synchronization between suppliers and freight forwarders systems, causing over- or under-capacity planning whenever a plant’s material demands change abruptly, leading to higher logistics transportation costs. 

The goal of this project is to update the code base using Python. Additionally, a benchmark of the latest forecasting algorithms will be done. The python ecosystem for forecasting has grown a lot in the last few years. There are Python packages like [nitxla](https://nixtlaverse.nixtla.io/), [lightgbm](https://lightgbm.readthedocs.io/en/latest/Python-API.html), [catboost](https://catboost.ai/), as well as LLM forecasting models like [chronos](https://huggingface.co/amazon/chronos-t5-large) that offer many additional features that at the time I built the system were only partially available in the R package [forecast](https://cran.r-project.org/package=forecast).

I wrote a paper last year about this system.  It can be found here [Forecasting System for Inbound Logistics Material Flows at an International Automotive Company](https://www.mdpi.com/2673-4591/39/1/75)

**Note**: All the data referring to Plant and Supplier IDs has been annonymized for analytics porpuses. 


## Problem Description

Material volume moves from the consolidation centers to the plants via the main legs in and a Area Forwarding-based Inbound Logistics Network, as explained in the following image:

![problem description](img/Area_Forwarding_based_inbound_logistics.png)

The idea is to create a forecasting system which is accurate and robust to adapt for outliers and unexpected events (for example COVID). To evaluate the forecast accuracy the `MAE (Mean Absolute Error)` and `SMAPE (Symmetric Mean Absolute Error)` will be used. This will allow us to care about the fact that in some months the transportation volume could have been 0. 

The test timeframes are: 
- Jan 2022 - Apr 2022
- May 2022 - Aug 2022
- Jul 2022 - Oct 2022

This means that models tested in each frame can only be trained with data prior to that frame to avoid data leakage. 

One of the main KPI's to track will be how many timeseries are in a particular `SMAPE` range, for that we will use the following intervals: 

- 0% to 10%
- 10 to 20%
- 20 to 30%
- 30 to 40%
- greater than 40%

The business experts are particularly interested in having a forecating systems for which most of the timeseries have a `SMAPE` of less or equal than 20%.

## Solution Approach

The approach to solve the problem is:

- (1) Analyze the data and create some graphs to explore the problem at hand. Create an Exploratory data analysis.
- (2) Create a medallion architecture to track data quality: Bronze layer for raw data, Silver layer for cleaned data, Gold layer for aggregated or feature-engineered data. This is a common approach to model data in machine learning problems. Ref: [ Medallion Architecture](https://www.databricks.com/glossary/medallion-architecture) 
- (3) Train different models and store their accuracy results in the database. We will train different models to cover the full range of current forecasting algorithms:
    - Statistical Models
    - Machine Learning Models
    - Deep Learning Models
    - LLM Timeseries Forecasting Models
- (4) Evaluate all the results, define the best performing models, and write up the lessons learned and possible next steps. 


This project contains the following sections:


```{tableofcontents}
```