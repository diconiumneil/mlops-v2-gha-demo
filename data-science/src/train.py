# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Trains ML model using training dataset. Saves trained model.
"""

import argparse

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import mlflow
import mlflow.sklearn

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

    parser = argparse.ArgumentParser("train")
    parser.add_argument("--train_data", type=str, help="Path to train dataset")
    parser.add_argument("--model_output", type=str, help="Path of output model")

    # classifier specific arguments
    parser.add_argument('--regressor__n_estimators', type=int, default=500,
                        help='Number of trees')
    parser.add_argument('--regressor__bootstrap', type=int, default=1,
                        help='Method of selecting samples for training each tree')
    parser.add_argument('--regressor__max_depth', type=int, default=10,
                        help=' Maximum number of levels in tree')
    parser.add_argument('--regressor__max_features', type=str, default='auto',
                        help='Number of features to consider at every split')
    parser.add_argument('--regressor__min_samples_leaf', type=int, default=4,
                        help='Minimum number of samples required at each leaf node')
    parser.add_argument('--regressor__min_samples_split', type=int, default=5,
                        help='Minimum number of samples required to split a node')

    args = parser.parse_args()

    return args

def main(args):
    '''Read train dataset, train model, save trained model'''

    # Read train data
    train_data = pd.read_parquet(Path(args.train_data))

    # Split the data into input(X) and output(y)
    y_train = train_data[TARGET_COL]
    X_train = train_data[NUMERIC_COLS]

    # Train a Random Forest Regression Model with the training set
    model = RandomForestRegressor(n_estimators = args.regressor__n_estimators,
                                  bootstrap = args.regressor__bootstrap,
                                  max_depth = args.regressor__max_depth,
                                  max_features = args.regressor__max_features,
                                  min_samples_leaf = args.regressor__min_samples_leaf,
                                  min_samples_split = args.regressor__min_samples_split,
                                  random_state=0)

    # log model hyperparameters
    mlflow.log_param("model", "RandomForestRegressor")
    mlflow.log_param("n_estimators", args.regressor__n_estimators)
    mlflow.log_param("bootstrap", args.regressor__bootstrap)
    mlflow.log_param("max_depth", args.regressor__max_depth)
    mlflow.log_param("max_features", args.regressor__max_features)
    mlflow.log_param("min_samples_leaf", args.regressor__min_samples_leaf)
    mlflow.log_param("min_samples_split", args.regressor__min_samples_split)

    # Train model with the train set
    model.fit(X_train, y_train)

    # Predict using the Regression Model
    yhat_train = model.predict(X_train)

    # Evaluate Regression performance with the train set
    r2 = r2_score(y_train, yhat_train)
    mse = mean_squared_error(y_train, yhat_train)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_train, yhat_train)
    
    # log model performance metrics
    mlflow.log_metric("train r2", r2)
    mlflow.log_metric("train mse", mse)
    mlflow.log_metric("train rmse", rmse)
    mlflow.log_metric("train mae", mae)

    # Visualize results
    plt.scatter(y_train, yhat_train,  color='black')
    plt.plot(y_train, y_train, color='blue', linewidth=3)
    plt.xlabel("Real value")
    plt.ylabel("Predicted value")
    plt.savefig("regression_results.png")
    mlflow.log_artifact("regression_results.png")

    # Save the model
    mlflow.sklearn.save_model(sk_model=model, path=args.model_output)


if __name__ == "__main__":
    
    mlflow.start_run()

    # ---------- Parse Arguments ----------- #
    # -------------------------------------- #

    args = parse_args()

    lines = [
        f"Train dataset input path: {args.train_data}",
        f"Model output path: {args.model_output}",
        f"n_estimators: {args.regressor__n_estimators}",
        f"bootstrap: {args.regressor__bootstrap}",
        f"max_depth: {args.regressor__max_depth}",
        f"max_features: {args.regressor__max_features}",
        f"min_samples_leaf: {args.regressor__min_samples_leaf}",
        f"min_samples_split: {args.regressor__min_samples_split}"
    ]

    for line in lines:
        print(line)

    main(args)

    mlflow.end_run()
    