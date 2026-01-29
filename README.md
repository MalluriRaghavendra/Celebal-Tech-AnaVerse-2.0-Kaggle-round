# Celebal-Tech-AnaVerse-2.0-Kaggle-round
🔍 Sensor Anomaly Detection using LightGBM
📌 Project Overview

This project focuses on building a high-performance machine learning pipeline to detect anomalies in sensor data collected from an energy manufacturing plant. The goal is to accurately classify whether a given sensor reading represents normal behavior (0) or an anomaly (1).

The solution is designed for Kaggle-style machine learning challenges and demonstrates an end-to-end workflow including data preprocessing, feature engineering, model training, threshold optimization, and automated submission file generation.

🎯 Objective

To predict anomalies based on time-series sensor readings using efficient feature engineering and a powerful gradient boosting model (LightGBM), optimized for F1-score performance.

📂 Dataset Description

The dataset contains sensor readings captured at fixed time intervals from an energy manufacturing plant.

Main Columns:

Date – Timestamp of the sensor reading

X1 to X5 – Sensor values

target – Anomaly label (0 = Normal, 1 = Anomaly)

Files used:

train.parquet – Training data (with target)

test.parquet – Test data (without target)

⚙️ Tech Stack

Python

Pandas, NumPy

LightGBM

Scikit-learn

Google Colab / Kaggle Notebook

🧠 Approach

Feature Engineering

Extracted time-based features from the Date column (year, month, day, weekday, hour)

Created cyclical features using sine and cosine transformations

Preprocessing

Removed unnecessary columns

Stratified train-validation split

Modeling

Trained a LightGBM binary classifier

Used early stopping for faster and stable training

Handled class imbalance using class_weight="balanced"

Evaluation

Optimized the classification threshold based on F1-score

Prediction

Generated predictions on test data

Exported results to submission.csv

🚀 How to Run
1️⃣ Install dependencies (Colab / Local)
pip install lightgbm xgboost catboost pyarrow fastparquet

2️⃣ Upload datasets

Place the following files in your working directory:

train.parquet

test.parquet

3️⃣ Run the notebook cells in order

The notebook will:

Train the model

Validate performance

Automatically generate submission.csv

📈 Results

The final model is optimized for anomaly detection and achieves strong performance on validation data. Threshold tuning significantly improves F1-score compared to using the default 0.5 cutoff.

📁 Output

submission.csv – Final predictions file ready for Kaggle submission

🧑‍💻 Author

Malluri Raghavendra
Computer Science Student, SR University

Portfolio: (Add your portfolio link here)
LinkedIn / GitHub: (Optional)

🏷️ Keywords

Anomaly Detection Machine Learning LightGBM Sensor Data Kaggle Challenge Time Series Features F1 Score Optimization
