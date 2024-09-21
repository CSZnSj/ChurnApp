# training.py

import sys
import os
sys.path.append('/home/sajjad/Projects/ChurnApp')

from pyspark.sql import SparkSession, DataFrame
from typing import Any, Dict
from src.logger import setup_logger
from src.utils import *
from src.model_utils import *

# Set up logger
logger = setup_logger(__name__)

def main(
        model_name: str, 
        metric_name: str) -> None:
    """
    Main function for the model training process.

    Args:
        model_name (str): The name of the machine learning model to train (e.g., 'gbt', 'rf').
        metric_name (str): The evaluation metric to use for model selection (e.g., 'f1', 'accuracy').

    Steps:
        1. Load configuration from the config file.
        2. Check if the model already exists; if so, skip training.
        3. Create a Spark session.
        4. Load preprocessed training data.
        5. Initialize the classifier and parameter grid for hyperparameter tuning.
        6. Define the evaluator for model evaluation.
        7. Train the model using cross-validation.
        8. Evaluate the model on the training data.
        9. Save the trained model to the specified output path.
    """

    logger.info(f"Starting training with model: {model_name}, metric: {metric_name}")
    
    spark = None  # Initialize spark session variable
    try:
        # Step 1: Load configuration
        config = load_config("config.json")
        
        # Retrieve input and output paths from config
        preprocessed_dataset_input_path = get_config_value(config, "preprocessed", "path").format(type="train")
        model_output_path = get_config_value(config, "model", "path").format(name=f"{model_name}__{metric_name}")

        # Step 2: Check if the model already exists
        if os.path.exists(model_output_path):
            logger.info(f"Model already exists at {model_output_path}. Skipping training.")
            return
        
        # Step 3: Create Spark session
        spark = create_spark_session(app_name="Model Training")
        spark.conf.set("spark.sql.parquet.datetimeRebaseModeInWrite", "CORRECTED")

        # Step 4: Load training data
        train_df = read_parquet(spark=spark, parquet_path=preprocessed_dataset_input_path)

        # Step 5: Define the classifier and parameter grid
        logger.info(f"Step 4: Initializing classifier and parameter grid for {model_name}.")
        classifier = get_classifier(model_name=model_name)
        param_grid = get_param_grid(model_name=model_name)

        # Step 6: Define the evaluator for model performance measurement
        logger.info(f"Step 5: Defining evaluator based on the {metric_name} metric.")
        evaluator = get_evaluator(metric_name=metric_name)

        # Step 7: Train the model using the training data
        logger.info("Step 6: Training the model using cross-validation.")
        model = train_model(model=classifier, df=train_df, evaluator=evaluator, param_grid=param_grid)

        # Step 8: Evaluate the model
        logger.info("Step 7: Evaluating the model on the training data.")
        evaluate_model(model=model, df=train_df, evaluator=evaluator, metric_name=metric_name)

        # Step 9: Save the trained model to the specified output path
        logger.info(f"Step 8: Saving the trained model to {model_output_path}.")
        save_model(model=model, output_path=model_output_path)

        logger.info(f" Training model: {model_name}, metric: {metric_name} is finished successfully")

    except Exception as e:
        # Catch and log any exception that occurs during the process
        logger.error(f"An error occurred during the training process: {str(e)}")
        raise
    finally:
        # Ensure the Spark session is stopped to release resources
        if spark:
            logger.info("Stopping Spark session.")
            spark.stop()
            logger.info("Spark session stopped.")

if __name__ == '__main__':
    main()