# ingegst.py
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import (
    StructType, StructField, StringType, LongType, DoubleType, DecimalType
)
from src.logger import setup_logger
from src.utils import *
class CustomSchema:
    """Defines custom schemas for different datasets."""
    
    loan_assign = StructType([
        StructField("bib_id", StringType(), True),
        StructField("date_key", StringType(), True),  # To be converted to date
        StructField("fake_id", StringType(), True),
        StructField("nid_hash", StringType(), True),
        StructField("loan_id", StringType(), True),
        StructField("loan_amount", LongType(), True),
        StructField("date_timestamp", StringType(), True)  # To be converted to timestamp
    ])

    loan_recovery = StructType([
        StructField("bib_id", StringType(), True),
        StructField("date_key", StringType(), True),  # To be converted to date
        StructField("fake_id", StringType(), True),
        StructField("nid_hash", StringType(), True),
        StructField("loan_id", StringType(), True),
        StructField("loan_amount", LongType(), True),
        StructField("hsdp_recovery", LongType(), True),
        StructField("date_timestamp", StringType(), True)  # To be converted to timestamp
    ])

    package = StructType([
        StructField("bib_id", StringType(), True),
        StructField("date_key", StringType(), True),  # To be converted to date
        StructField("fake_id", StringType(), True),
        StructField("nid_hash", StringType(), True),
        StructField("offering_code", StringType(), True),
        StructField("offer_amount", LongType(), True),
        StructField("offering_name", StringType(), True),
        StructField("activation_date", StringType(), True),  # To be converted to timestamp
        StructField("deactivation_date", StringType(), True)  # To be converted to timestamp
    ])

    recharge = StructType([
        StructField("bib_id", StringType(), True),
        StructField("date_key", StringType(), True),  # To be converted to date
        StructField("fake_id", StringType(), True),
        StructField("nid_hash", StringType(), True),
        StructField("recharge_value_amt", LongType(), True),
        StructField("recharge_dt", StringType(), True),  # To be converted to timestamp
        StructField("origin_host_nm", StringType(), True),
        StructField("account_balance_before_amt", LongType(), True),
        StructField("account_balance_after_amt", LongType(), True)
    ])

    cdr = StructType([
        StructField("bib_id", StringType(), True),
        StructField("date_key", StringType(), True),  # To be converted to date
        StructField("fake_id", StringType(), True),
        StructField("nid_hash", StringType(), True),
        StructField("sms_count", LongType(), True),
        StructField("voice_count", LongType(), True),
        StructField("call_duration", LongType(), True),
        StructField("gprs_usage", LongType(), True),
        StructField("voice_session_cos", DecimalType(10, 6), True)
    ])

    user = StructType([
        StructField("bib_id", StringType(), True),
        StructField("fake_id", StringType(), True),
        StructField("nid_hash", StringType(), True),
        StructField("contract_type_v", StringType(), True),
        StructField("gender_v", StringType(), True),
        StructField("registration_date_d", StringType(), True),  # To be converted to date
        StructField("date_of_birth_d", StringType(), True),  # To be converted to date
        StructField("ability_status", StringType(), True),
        StructField("account_balance", DoubleType(), True),
        StructField("base_station_cd", StringType(), True),
        StructField("sitei", StringType(), True)
    ])

# Initialize logger
logger = setup_logger(__name__)

def ingest_data(spark: SparkSession, config: dict) -> None:
    """
    Orchestrates the data ingestion process from CSV to Parquet format.

    :param spark: SparkSession instance used for the data ingestion process.
    :param config: Configuration dictionary containing paths and column info.
    """
    logger.info("Starting data ingestion process...")
    try:
        # Retrieve values from config using get_config_value to handle errors if keys are missing
        months = get_config_value(config, "months")
        keys = get_config_value(config, "keys")

        for month in months:
            for key in keys:
                # Get the CSV and Parquet paths from the config using get_config_value
                csv_path = get_config_value(config, "ingestion", "paths", "csv", key).format(month=month)
                parquet_path = get_config_value(config, "ingestion", "paths", "parquet", key).format(month=month)

                # Get the columns for timestamp and date conversion
                timestamp_columns = get_config_value(config, "ingestion", "convert_date_columns", key, "timestamp_columns")
                date_columns = get_config_value(config, "ingestion", "convert_date_columns", key, "date_columns")

                # Read CSV file into a DataFrame
                df = read_csv_with_schema(spark=spark, csv_path=csv_path, schema=getattr(CustomSchema, key))

                # Convert timestamp and date columns
                df = convert_date_columns(df=df, csv_path=csv_path, timestamp_columns=timestamp_columns, date_columns=date_columns)

                # Write the DataFrame to Parquet
                write_parquet(df, parquet_path, mode="overwrite")

                logger.info(f"Successfully processed and saved data for key {key} for month {month}.")

    except KeyError as e:
        logger.error(f"Missing configuration key: {e}")
        raise
    except Exception as e:
        logger.error(f"Error during data ingestion: {e}")
        raise

def main():
    """
    Main function to orchestrate the entire data ingestion process.
    Handles Spark session creation and cleanup.
    """
    spark = None
    try:
        # Load configuration
        config = load_config("config.json")

        # Create Spark session
        spark = create_spark_session("DataIngestion")
        spark.conf.set("spark.sql.parquet.datetimeRebaseModeInWrite", "CORRECTED")

        # Call the ingestion function with the Spark session and configuration
        ingest_data(spark, config)

    except Exception as e:
        logger.error(f"Failed to complete data ingestion: {e}")
        raise
    finally:
        # Stop Spark session if it was started
        if spark:
            spark.stop()
            logger.info("Spark session stopped.")

if __name__ == "__main__":
    main()