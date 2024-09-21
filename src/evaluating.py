# evaluating.py

import sys
sys.path.append('/home/sajjad/Projects/ChurnApp')

from pyspark.sql import SparkSession
from src.logger import setup_logger
from src.utils import *
from src.model_utils import load_model, get_evaluator, evaluate_model

logger = setup_logger(__name__)

def main(
        model_name: str = "gbt", 
        metric_name: str = "f1") -> None:
    """
    Main function for evaluating a trained machine learning model on eval data.

    Args:
        model_name (str): The name of the trained machine learning model to evaluate (e.g., 'gbt', 'rf').
        metric_name (str): The evaluation metric to use for performance assessment (e.g., 'f1', 'accuracy').

    Steps:
        1. Load configuration from the config file.
        2. Create a Spark session.
        3. Load preprocessed eval data.
        4. Load the trained model from the specified path.
        5. Define the evaluator for model performance measurement.
        6. Evaluate the model on the eval data.
    """

    logger.info(f"Starting evaluation with model: {model_name}, metric: {metric_name}")

    spark = None  # Initialize Spark session variable
    try:
        # Step 1: Load configuration
        logger.info("Step 1: Loading configuration.")
        config = load_config("config.json")
        
        # Retrieve input paths and model paths from the config file
        preprocessed_dataset_input_path = get_config_value(config, "preprocessed", "path").format(type="eval")
        trained_model_path = get_config_value(config, "model", "path").format(name=f"{model_name}__{metric_name}")

        # Step 2: Create Spark session
        logger.info("Step 2: Creating Spark session.")
        spark = create_spark_session(app_name="Model Evaluation")
        spark.conf.set("spark.sql.parquet.datetimeRebaseModeInWrite", "CORRECTED")

        # Step 3: Load eval data
        eval_df = read_parquet(spark=spark, parquet_path=preprocessed_dataset_input_path)

        # Step 4: Load the trained model
        model = load_model(model_name=model_name, model_path=trained_model_path)

        # Step 5: Define the evaluator for model performance measurement
        logger.info(f"Step 5: Defining evaluator based on the {metric_name} metric.")
        evaluator = get_evaluator(metric_name=metric_name)

        # Step 6: Evaluate the model on the eval data
        logger.info(f"Step 6: Evaluating the model using the {metric_name} metric.")
        evaluate_model(model=model, df=eval_df, evaluator=evaluator, metric_name=metric_name)

        logger.info(f"Evaluation model: {model_name}, metric: {metric_name} is finished successfully")

    except Exception as e:
        # Catch and log any exceptions during the process
        logger.error(f"An error occurred during the evaluation process: {str(e)}")
        raise
    finally:
        # Ensure Spark session is stopped to release resources
        if spark:
            logger.info("Stopping Spark session.")
            spark.stop()
            logger.info("Spark session stopped.")

if __name__ == '__main__':
    main()