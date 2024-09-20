# evaluating.py

import sys
sys.path.append('/home/sajjad/Projects/ChurnApp')

from pyspark.sql import SparkSession
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import RandomForestClassificationModel
from src.logger import setup_logger
from src.utils import load_config, get_config_value, create_spark_session, read_parquet, write_parquet 

logger = setup_logger(__name__)

def load_model(spark: SparkSession, model_path: str):
    """Load any trained model."""
    try:
        logger.info(f"Loading model from {model_path}")
        model = RandomForestClassificationModel.load(model_path)
        logger.info(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def evaluate_model(model, df, label_col: str):
    """Evaluate the model using accuracy metric."""
    try:
        logger.info("Evaluating model performance...")
        predictions = model.transform(df)
        evaluator = MulticlassClassificationEvaluator(labelCol=label_col, metricName="f1")
        accuracy = evaluator.evaluate(predictions)
        logger.info(f"Model Accuracy: {accuracy:.4f}")
        return accuracy
    except Exception as e:
        logger.error(f"Error during model evaluation: {str(e)}")
        raise

if __name__ == '__main__':
    # Set up paths and variables
    label_col = "label"
    model_name = "random_forest_01"

    # Step 1: Load configuration
    config = load_config("config.json")
    preprocessed_dataset_input_path = get_config_value(config, "preprocessed", "path").format(type="test")
    trained_model_path = get_config_value(config, "model", "path").format(name=model_name)
    
    # Create Spark session
    spark = create_spark_session("RandomForestEvaluation")
    spark.conf.set("spark.sql.parquet.datetimeRebaseModeInWrite", "CORRECTED")

    # Load test data
    test_df = read_parquet(spark=spark, parquet_path=preprocessed_dataset_input_path)

    # Load model
    model = load_model(spark, trained_model_path)

    # Evaluate the model on test data
    evaluate_model(model, test_df, label_col)

    # Stop Spark session
    spark.stop()