# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Prepares raw data and provides training, validation and test datasets
"""

import argparse

from pathlib import Path
import os
import numpy as np
import pandas as pd

import mlflow

TARGET_COL = "Units Sold"

NUMERIC_COLS = ['Inventory Level', 'Units Sold', 'Units Ordered', 'Price', 'Discount',
       'Competitor Pricing', 'day_of_week', 'month', 'day_of_month',
       'Weather Condition_Cloudy', 'Weather Condition_Rainy',
       'Weather Condition_Snowy', 'Weather Condition_Sunny',
       'Holiday/Promotion_0', 'Holiday/Promotion_1', 'Seasonality_Autumn',
       'Seasonality_Spring', 'Seasonality_Summer', 'Seasonality_Winter',
       'Store ID_S001', 'Store ID_S002', 'Store ID_S003', 'Store ID_S004',
       'Store ID_S005', 'Units_Sold_Lag_1', 'Units_Sold_Lag_2',
       'Units_Sold_Lag_3', 'Units_Sold_Lag_4', 'Units_Sold_Lag_5',
       'Units_Sold_Lag_6', 'Units_Sold_Lag_7', 'Units_Sold_Lag_8',
       'Units_Sold_Lag_9', 'Units_Sold_Lag_10', 'Units_Sold_Lag_11',
       'Units_Sold_Lag_12', 'Units_Sold_Lag_13', 'Units_Sold_Lag_14',
       'Units_Sold_Lag_15', 'Units_Sold_Lag_16', 'Units_Sold_Lag_17',
       'Units_Sold_Lag_18', 'Units_Sold_Lag_19', 'Units_Sold_Lag_20',
       'Units_Sold_Lag_21', 'Units_Sold_Lag_22', 'Units_Sold_Lag_23',
       'Units_Sold_Lag_24', 'Units_Sold_Lag_25', 'Units_Sold_Lag_26',
       'Units_Sold_Lag_27']

def parse_args():
    '''Parse input arguments'''

    parser = argparse.ArgumentParser("prep")
    parser.add_argument("--raw_data", type=str, help="Path to raw data")
    parser.add_argument("--train_data", type=str, help="Path to train dataset")
    parser.add_argument("--val_data", type=str, help="Path to test dataset")
    parser.add_argument("--test_data", type=str, help="Path to test dataset")
    
    parser.add_argument("--enable_monitoring", type=str, help="enable logging to ADX")
    parser.add_argument("--table_name", type=str, default="mlmonitoring", help="Table name in ADX for logging")
    
    args = parser.parse_args()

    return args

def log_training_data(df, table_name):
    from obs.collector import Online_Collector
    collector = Online_Collector(table_name)
    collector.batch_collect(df)

def create_features(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Function that prepares the data for training
    
    Args: 
        data: pd.DataFrame: Raw data
    
    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Train, validation and test datasets
    """
    
    data_sample = data[
        (data["Product ID"] == "P0001")
    ].copy()

    
    data_sample.loc[:, "day_of_week"] = pd.to_datetime(data_sample["Date"]).dt.dayofweek
    data_sample.loc[:, "month"] = pd.to_datetime(data_sample["Date"]).dt.month
    data_sample.loc[:, "day_of_month"] = pd.to_datetime(data_sample["Date"]).dt.day

    data_sample.loc[:, "competitor_price_ratio"] = data_sample["Price"] / data_sample["Competitor Pricing"]

    data_sample = pd.get_dummies(data_sample, columns=["Weather Condition", "Holiday/Promotion", "Seasonality", "Store ID"])
    data_sample.drop(["Region", "Product ID", "Category", "Demand Forecast", "Date"], axis=1, inplace=True)

    for lag in range(1, 28):
        data_sample[f"Units_Sold_Lag_{lag}"] = data_sample["Units Sold"].shift(lag)
    data_sample.dropna(inplace=True)

    data_sample = data_sample[NUMERIC_COLS + [TARGET_COL]]
    train = data_sample.iloc[:int(0.7*len(data_sample)), :]
    val = data_sample.iloc[int(0.7*len(data_sample)):int(0.85*len(data_sample)), :]
    test = data_sample.iloc[int(0.85*len(data_sample)):, :]

    return train, val, test

def main(args):
    '''Read, split, and save datasets'''

    # ------------ Reading Data ------------ #
    # -------------------------------------- #

    data = pd.read_csv((Path(args.raw_data)))
    

    # ------------- Split Data ------------- #
    # -------------------------------------- #

    # Split data into train, val and test datasets


    train, val, test = create_features(data)

    mlflow.log_metric('train size', train.shape[0])
    mlflow.log_metric('val size', val.shape[0])
    mlflow.log_metric('test size', test.shape[0])

    train.to_parquet((Path(args.train_data) / "train.parquet"))
    val.to_parquet((Path(args.val_data) / "val.parquet"))
    test.to_parquet((Path(args.test_data) / "test.parquet"))

    if (args.enable_monitoring.lower() == 'true' or args.enable_monitoring == '1' or args.enable_monitoring.lower() == 'yes'):
        log_training_data(data, args.table_name)


if __name__ == "__main__":

    mlflow.start_run()

    # ---------- Parse Arguments ----------- #
    # -------------------------------------- #

    args = parse_args()

    lines = [
        f"Raw data path: {args.raw_data}",
        f"Train dataset output path: {args.train_data}",
        f"Val dataset output path: {args.val_data}",
        f"Test dataset path: {args.test_data}",

    ]

    for line in lines:
        print(line)
    
    main(args)

    mlflow.end_run()

    
