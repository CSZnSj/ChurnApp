import json
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import StructType
from pyspark.sql.utils import AnalysisException
from pyspark.sql.functions import to_date, to_timestamp
from src.logger import setup_logger

# Initialize logger
logger = setup_logger(__name__)

def load_config(
        config_path: str = "config.json") -> dict:
    """
    Loads the configuration from a JSON file.

    :param config_path: parquet_path to the JSON configuration file (default is 'config.json').
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

def get_config_value(
        config: dict, 
        *keys: str):
    """
    Retrieves a nested value from a configuration dictionary using a variable number of keys.
    
    This function allows you to access deeply nested values in a dictionary based on a dynamic 
    list of keys. It handles missing keys by logging an error and raising a KeyError.

    Example usage:
        get_config_value(config, "ingestion", "paths", "csv", "loan_assign")

    :param config: The configuration dictionary to search.
    :param keys: Variable-length list of keys to access the nested value.
    :return: The value from the config if all keys are found.
    :raises KeyError: If any of the provided keys do not exist in the dictionary.
    """
    try:
        # Start with the entire configuration and narrow down based on the provided keys
        value = config
        for key in keys:
            value = value[key]  # Traverse through the dictionary using the keys
            
        logger.info(f"Successfully retrieved config value for keys: {keys}")
        return value
    
    except KeyError as e:
        logger.error(f"Missing required key {e} in configuration: {' -> '.join(keys)}")
        raise KeyError(f"Configuration key not found for: {' -> '.join(keys)}") from e
    except Exception as e:
        logger.error(f"Unexpected error retrieving configuration value for keys {keys}: {e}")
        raise

def create_spark_session(
        app_name: str) -> SparkSession:
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

def read_csv_with_schema(
        spark: SparkSession, 
        csv_path: str, 
        schema: StructType) -> DataFrame:
    """
    Reads a CSV file into a Spark DataFrame using the provided schema.

    This function reads a CSV file from the specified parquet_path into a Spark DataFrame
    with a user-defined schema, ensuring the correct data types are enforced.

    :param spark: SparkSession instance used for reading the CSV file.
    :param parquet_path: path to the CSV file.
    :param schema: A StructType object defining the schema of the CSV.
    :return: DataFrame containing the data from the CSV file.
    :raises: Exception if the file cannot be read or the schema is invalid.
    """
    try:
        logger.info(f"Attempting to read CSV file: {csv_path} with provided schema")
        df = spark.read.csv(csv_path, header=True, schema=schema)
        logger.info(f"Successfully read CSV file from: {csv_path}")
        return df
    except FileNotFoundError:
        logger.error(f"CSV file not found at: {csv_path}")
        raise
    except Exception as e:
        logger.error(f"Error occurred while reading CSV file at {csv_path}. Error: {e}")
        raise

def convert_date_columns(
    df: DataFrame, 
    csv_path: str, 
    timestamp_columns: list = None, 
    date_columns: list = None
) -> DataFrame:
    """
    Converts specified columns in a DataFrame to date or timestamp types.

    This function processes the specified timestamp and date columns in a DataFrame and converts them
    into `timestamp` and `date` data types respectively. If a column is not found in the DataFrame, it logs a warning
    and skips the conversion. 

    :param df: The input DataFrame to be processed.
    :param csv_path: Path of the CSV file from which the DataFrame was read, used for logging purposes.
    :param timestamp_columns: List of columns to be converted to `timestamp` type. Default is an empty list.
    :param date_columns: List of columns to be converted to `date` type. Default is `["date_key"]`.
    
    :return: A DataFrame with the converted date and timestamp columns.
    
    :raises AnalysisException: If a column conversion fails due to invalid format.
    :raises Exception: For any other unexpected errors during the conversion process.
    """
    
    # Set default values for columns if not provided
    timestamp_columns = timestamp_columns or []
    date_columns = date_columns or ["date_key"]

    def convert_column(col_name: str, format_str: str, conversion_func) -> DataFrame:
        """
        Helper function to convert a column to the specified format using a provided function.

        :param col_name: The name of the column to be converted.
        :param format_str: The format string for the conversion (e.g., 'yyyyMMdd HH:mm:ss').
        :param conversion_func: The function to apply for conversion (e.g., `to_timestamp` or `to_date`).
        :return: DataFrame with the converted column.
        """
        if col_name in df.columns:
            df_with_converted_column = df.withColumn(col_name, conversion_func(col_name, format_str))
            logger.info(f"Successfully converted column '{col_name}' in {csv_path} to {conversion_func.__name__}.")
            return df_with_converted_column
        else:
            logger.warning(f"Column '{col_name}' not found in {csv_path}. Skipping conversion.")
            return df
    
    try:
        # Convert timestamp columns
        for col in timestamp_columns:
            df = convert_column(col, "yyyyMMdd HH:mm:ss", to_timestamp)
        
        # Convert date columns
        for col in date_columns:
            df = convert_column(col, "yyyyMMdd", to_date)
        
        return df

    except AnalysisException as ae:
        logger.error(f"Failed to convert date/timestamp columns in {csv_path} due to an analysis error: {ae}")
        raise

    except Exception as e:
        logger.exception(f"Unexpected error during date/timestamp conversion in {csv_path}: {e}")
        raise

def write_parquet(df: DataFrame, output_parquet_path: str, mode: str = "overwrite") -> None:
    """
    Writes a DataFrame to a Parquet file.

    :param df: The DataFrame to write.
    :param output_parquet_path: The output parquet_path where the Parquet file will be saved.
    :param mode: Save mode (default is 'overwrite'). Options: 'append', 'overwrite', 'ignore', 'error'.
    :raises: Exception if writing the Parquet file fails.
    """
    try:
        df.write.mode(mode).parquet(output_parquet_path)
        logger.info(f"DataFrame successfully written to Parquet at: {output_parquet_path} with mode: {mode}")
    except Exception as e:
        logger.error(f"Failed to write DataFrame to Parquet at {output_parquet_path}. Error: {e}")
        raise

def read_parquet(spark: SparkSession, parquet_path: str) -> DataFrame:
    """
    Reads a Parquet file into a Spark DataFrame.

    This function reads data from a Parquet file located at the specified parquet_path and
    loads it into a Spark DataFrame. It ensures proper error handling and logging
    for troubleshooting.

    :param spark: The SparkSession instance used for reading the Parquet file.
    :param parquet_path: path to the Parquet file.
    :return: DataFrame containing the data from the Parquet file.
    :raises: FileNotFoundError if the file does not exist.
             AnalysisException for issues related to reading the Parquet format.
             Exception for any other unexpected errors.
    """
    try:
        logger.info(f"Attempting to read Parquet file from: {parquet_path}")
        df = spark.read.parquet(parquet_path)
        logger.info(f"Successfully read Parquet file from: {parquet_path}")
        return df
    except AnalysisException as ae:
        logger.error(f"Error reading Parquet file at {parquet_path}. Issue with file format or parquet_path. Error: {ae}")
        raise
    except FileNotFoundError:
        logger.error(f"Parquet file not found at: {parquet_path}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error occurred while reading Parquet file at {parquet_path}. Error: {e}")
        raise
