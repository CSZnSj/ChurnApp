import json
from pyspark.sql import SparkSession
import logging

def create_spark_session(app_name: str) -> SparkSession:
    """
    Creates and returns a SparkSession with the specified application name.

    :param app_name: Name of the Spark application.
    :return: SparkSession object.
    """
    try:
        spark = SparkSession.builder.appName(app_name).getOrCreate()
        logging.info(f"Spark session created for app: {app_name}")
        return spark
    except Exception as e:
        logging.error(f"Failed to create Spark session for app: {app_name}. Error: {e}")
        raise

def load_config(config_path: str = "config.json") -> dict:
    """
    Loads the configuration from a JSON file.

    :param config_path: Path to the JSON configuration file (default is 'config.json').
    :return: Configuration dictionary.
    """
    try:
        with open(config_path, 'r') as file:
            config = json.load(file)
        logging.info(f"Successfully loaded configuration from {config_path}")
        return config
    except FileNotFoundError:
        logging.error(f"Configuration file not found: {config_path}")
        raise
    except json.JSONDecodeError as e:
        logging.error(f"Error parsing JSON from {config_path}: {e}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error loading config from {config_path}: {e}")
        raise