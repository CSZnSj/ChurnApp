import json
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import StructType
from pyspark.sql.utils import AnalysisException
from pyspark.sql.functions import to_date, to_timestamp
from src.logger import setup_logger

# Initialize logger
logger = setup_logger(__name__)

def load_config(config_path: str = "config.json") -> dict:
    """
    Loads the configuration from a JSON file.

    :param config_path: Path to the JSON configuration file (default is 'config.json').
    :return: Configuration dictionary.
    :raises: FileNotFoundError if the configuration file is not found.
             JSONDecodeError if the file is not a valid JSON.
             Exception for any other unexpected error.
    """
    try:
        with open(config_path, 'r') as file:
            config = json.load(file)
        logger.info(f"Successfully loaded configuration from {config_path}")
        return config
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing JSON from {config_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading config from {config_path}: {e}")
        raise

def create_spark_session(app_name: str) -> SparkSession:
    """
    Creates and returns a SparkSession with the specified application name.

    :param app_name: Name of the Spark application.
    :return: SparkSession object.
    :raises: Exception if Spark session creation fails.
    """
    try:
        spark = SparkSession.builder.appName(app_name).getOrCreate()
        logger.info(f"Spark session created for app: {app_name}")
        return spark
    except Exception as e:
        logger.error(f"Failed to create Spark session for app: {app_name}. Error: {e}")
        raise

def read_csv_with_schema(spark: SparkSession, file_path: str, schema: StructType) -> DataFrame:
    """
    Reads a CSV file into a Spark DataFrame using the provided schema.

    This function reads a CSV file from the specified path into a Spark DataFrame
    with a user-defined schema, ensuring the correct data types are enforced.

    :param spark: SparkSession instance used for reading the CSV file.
    :param file_path: Path to the CSV file.
    :param schema: A StructType object defining the schema of the CSV.
    :return: DataFrame containing the data from the CSV file.
    :raises: Exception if the file cannot be read or the schema is invalid.
    """
    try:
        logger.info(f"Attempting to read CSV file: {file_path} with provided schema")
        df = spark.read.csv(file_path, header=True, schema=schema)
        logger.info(f"Successfully read CSV file from: {file_path}")
        return df
    except FileNotFoundError:
        logger.error(f"CSV file not found at: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error occurred while reading CSV file at {file_path}. Error: {e}")
        raise

def convert_date_columns(df: DataFrame, timestamp_columns=None, date_columns=None) -> DataFrame:
    """
    Converts specified columns to date or timestamp types.

    :param df: The input DataFrame.
    :param timestamp_columns: List of columns to be converted to timestamp type (default is None).
    :param date_columns: List of columns to be converted to date type (default is ["date_key"]).
    :return: DataFrame with the converted date/timestamp columns.
    :raises: Exception if column conversion fails.
    """
    if timestamp_columns is None:
        timestamp_columns = []
    if date_columns is None:
        date_columns = ["date_key"]

    try:
        # Convert timestamp columns
        for col in timestamp_columns:
            if col in df.columns:
                df = df.withColumn(col, to_timestamp(col, "yyyyMMdd HH:mm:ss"))
                logger.info(f"Converted column {col} to timestamp")

        # Convert date columns
        for col in date_columns:
            if col in df.columns:
                df = df.withColumn(col, to_date(col, "yyyyMMdd"))
                logger.info(f"Converted column {col} to date")

        return df
    except Exception as e:
        logger.error(f"Error converting date/timestamp columns. Error: {e}")
        raise

def write_parquet(df: DataFrame, output_path: str, mode: str = "overwrite") -> None:
    """
    Writes a DataFrame to a Parquet file.

    :param df: The DataFrame to write.
    :param output_path: The output path where the Parquet file will be saved.
    :param mode: Save mode (default is 'overwrite'). Options: 'append', 'overwrite', 'ignore', 'error'.
    :raises: Exception if writing the Parquet file fails.
    """
    try:
        df.write.mode(mode).parquet(output_path)
        logger.info(f"DataFrame successfully written to Parquet at: {output_path} with mode: {mode}")
    except Exception as e:
        logger.error(f"Failed to write DataFrame to Parquet at {output_path}. Error: {e}")
        raise

def read_parquet(spark: SparkSession, file_path: str) -> DataFrame:
    """
    Reads a Parquet file into a Spark DataFrame.

    This function reads data from a Parquet file located at the specified path and
    loads it into a Spark DataFrame. It ensures proper error handling and logging
    for troubleshooting.

    :param spark: The SparkSession instance used for reading the Parquet file.
    :param file_path: Path to the Parquet file.
    :return: DataFrame containing the data from the Parquet file.
    :raises: FileNotFoundError if the file does not exist.
             AnalysisException for issues related to reading the Parquet format.
             Exception for any other unexpected errors.
    """
    try:
        logger.info(f"Attempting to read Parquet file from: {file_path}")
        df = spark.read.parquet(file_path)
        logger.info(f"Successfully read Parquet file from: {file_path}")
        return df
    except AnalysisException as ae:
        logger.error(f"Error reading Parquet file at {file_path}. Issue with file format or path. Error: {ae}")
        raise
    except FileNotFoundError:
        logger.error(f"Parquet file not found at: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error occurred while reading Parquet file at {file_path}. Error: {e}")
        raise
