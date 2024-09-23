# deploying.py

import sys
sys.path.append('/home/sajjad/Projects/ChurnApp')

from pyspark.sql import SparkSession, DataFrame
from src.logger import setup_logger
from src.utils import load_config, get_config_value, create_spark_session, read_parquet, write_parquet 
from src.model_utils import load_model, save_model

import argparse

# Set up logger
logger = setup_logger(__name__)

def make_predictions(model, df: DataFrame) -> DataFrame:
    """
    Make predictions using the deployed model on the provided DataFrame.

    Args:
        model: Trained PySpark model.
        df (DataFrame): Preprocessed test DataFrame for making predictions.

    Returns:
        DataFrame: A DataFrame containing "bib_id" and "prediction" columns.
    """
    try:
        logger.info("Making predictions with the deployed model...")
        predictions = model.transform(df)
        return predictions.select("bib_id", "prediction")
    except Exception as e:
        logger.error(f"Error during predictions: {str(e)}")
        raise

def main(
        model_name: str, 
        metric_name: str) -> None:
    """
    Main function to deploy the model and generate predictions.

    Args:
        model_name (str): The name of the trained model to load.
        metric_name (str): The evaluation metric used to identify the best model.
    
    Workflow:
        - Load configuration from config.json.
        - Read the preprocessed test data from Parquet.
        - Load the trained model.
        - Generate predictions using the model.
        - Write predictions to the output location in Parquet format.
    """

    logger.info(f"Starting deployment with trained model: {model_name}, metric: {metric_name}")
    
    spark = None
    try:
        # Step 1: Load configuration
        config = load_config("config.json")
        data_path = get_config_value(config, "preprocessed", "path").format(type="test")
        name = f"{model_name}__{metric_name}"
        model_path = get_config_value(config, "model", "path").format(name=name)
        predictions_output_path = get_config_value(config, "predictions", "path").format(name=name)
        
        # Step 2: Create Spark session and read test data
        spark = create_spark_session("Deploying")
        df = read_parquet(spark, data_path)
        
        # Step 3: Load trained model
        logger.info(f"Loading model from {model_path}")
        model = load_model(model_name, model_path)
        
        # Step 4: Make predictions
        predictions = make_predictions(model, df)
        
        # Step 5: Write predictions to Parquet
        logger.info(f"Writing predictions to {predictions_output_path}")
        write_parquet(predictions, predictions_output_path)

        logger.info(f"Deployment trained model: {model_name}, metric: {metric_name} is finished successfully")

    except Exception as e:
        logger.error(f"Error in deployment process: {str(e)}")
        raise

    finally:
        # Ensure the Spark session is stopped to release resources
        if spark:
            logger.info("Stopping Spark session.")
            spark.stop()
            logger.info("Spark session stopped.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Deploy a trained machine learning model.")
    parser.add_argument('--model_name', type=str, required=True, help="The name of the model to deploy (e.g., 'gbt', 'rf').")
    parser.add_argument('--metric_name', type=str, required=True, help="The metric associated with the model (e.g., 'f1', 'accuracy').")

    args = parser.parse_args()

    main(model_name=args.model_name, metric_name=args.metric_name)