# ChurnApp: A Telecom Churn Prediction Project

## Overview
ChurnApp is a comprehensive data engineering and machine learning pipeline designed to predict customer churn in the telecom sector. It leverages Apache Spark and Airflow to process data, build machine learning models, and deploy them for real-world usage. The project is structured to handle large datasets efficiently, with a focus on scalability, reproducibility, and automation.

## Project Structure

```bash
ChurnApp/
│
├── data/
│     ├── raw/                       # Raw input data (CSV, Parquet files)
│     │     ├── csv/                 # Raw CSV data
│     │     ├── parquet/             # Raw Parquet data
│     │     
│     ├── intermediate/              # Intermediate data during processing
│     │     ├── label/               # Generated labels for the dataset
│     │     ├── dataset/             # Intermediate datasets after merging
│     │
│     ├── processed/                 # Final preprocessed data for model training/testing
│
├── airflow/
│       ├── dags/                    # Airflow DAG definitions for the pipeline
│       ├── logs/                    # Logs generated by Airflow tasks
│
├── src/                             # Source code for the project
│   ├── ingest.py                    # Script for data ingestion and conversion to Parquet
│   ├── prepare_label.py             # Script for preparing labels
│   ├── prepare_dataset.py           # Script for dataset preparation and merging PySpark DataFrames
│   ├── preprocess.py                # Script for preprocessing merged PySpark DataFrame and preparing it for model training
│   ├── training.py                  # Script for model training
│   ├── evaluating.py                # Script for model evaluation
│   ├── deploying.py                 # Script for model deployment
│   ├── logger.py                    # Custom logging functionality
│   ├── utils.py                     # Utility functions (Spark session creation, config loading, etc.)
│   ├── model_utils.py               # Model-specific utility functions (training, evaluation support)
│
├── results/                         # Output results from models and predictions
│    ├── models/                     # Trained model artifacts
│    ├── predictions/                # Generated model predictions for test data
│
├── notebooks/                       # Jupyter notebooks for data exploration and visualization
│        ├── plot.py                 # Scripts for visualization
│        ├── plot_tools.py           # Helper functions for visualization
│
├── requirements.txt                 # List of project dependencies
├── config.json                      # Defines the structure and paths for data ingestion, preprocessing, labeling, model storage, and predictions for the churn prediction project.
└── README.md                        # Project overview and setup instructions
```

## Features
- **Data Ingestion**: Automatically ingests raw data from CSV files and converts them to Parquet format for efficient storage and processing.
- **Label Preparation**: Generates churn labels based on customer behavior. Specifically, if a user purchases a package or recharges their account in a given month, but does not perform either action in the following month, the label is set to `1`, indicating that the user has churned (i.e., left the company). If the user continues to buy a package or recharge their account in subsequent months, the label is set to `0`, indicating no churn. This label is crucial for predicting future churn behavior and training the model to identify at-risk customers.
- **Dataset Preparation**: This script selects specified columns from the loan_assign, loan_recovery, package, and other relevant DataFrames, merging them into a single consolidated dataset. This ensures that the final dataset is streamlined for analysis and modeling, retaining only the necessary features while eliminating redundancy.
- **Preprocessing**: This script is designed to apply a series of transformations to the preprocessed train, evaluation, and test datasets for a churn prediction project. It includes techniques such as logarithmic transformations, scaling, capping, one-hot encoding, and handling missing values, ensuring that the data is appropriately prepared for model training and evaluation by selecting relevant features and transforming them into a suitable format.
- **Model Training**: Implements machine learning models using PySpark, including hyperparameter tuning and cross-validation to optimize performance.
- **Evaluation**: Evaluates models on eval data, providing key metrics such as accuracy, precision, recall, and F1-score to assess model performance.
- **Deployment**: Deploys trained models, enabling real-time predictions for customer churn in the telecom sector.
- **Logging & Monitoring**: Integrated with Airflow for task automation and logging, while using MLflow

# Setup Steps

1. Clone this repository:
    ```bash
    git clone https://github.com/CSZnSj/ChurnApp.git
    cd ChurnApp
    ```

2. Set up a Python virtual environment:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Initialize Airflow:
    ```bash
    airflow db init
    airflow scheduler
    airflow webserver --port 8080
    ```