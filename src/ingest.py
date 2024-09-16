from src.logger import setup_logger
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import (
    StructType, StructField, StringType, LongType, DoubleType, DecimalType
)
from pyspark.sql.functions import to_date, to_timestamp
from src.utils import create_spark_session, load_config

# Initialize logger
logger = setup_logger(__name__, log_file='./output/logs/ingest.log')

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


def read_csv_with_schema(spark: SparkSession, file_path: str, schema: StructType) -> DataFrame:
    """
    Reads a CSV file into a Spark DataFrame with the provided schema.
    """
    try:
        logger.info(f"Reading CSV file: {file_path}")
        df = spark.read.csv(file_path, header=True, schema=schema)
        logger.info(f"Successfully read CSV file: {file_path}")
        return df
    except Exception as e:
        logger.error(f"Error reading CSV file {file_path}: {e}")
        raise


def convert_date_columns(df: DataFrame, timestamp_columns=None, date_columns=None) -> DataFrame:
    """
    Converts specified columns to date or timestamp types.
    """
    if timestamp_columns is None:
        timestamp_columns = []
    if date_columns is None:
        date_columns = ["date_key"]

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


def ingest_data(config: dict) -> None:
    """
    Orchestrates the data ingestion process from CSV to Parquet format.
    """
    logger.info("Starting data ingestion process...")
    spark = None
    try:
        spark = create_spark_session("DataIngestion")
        spark.conf.set("spark.sql.parquet.datetimeRebaseModeInWrite", "CORRECTED")
        
        months = config.get("months", [])
        keys = config.get("keys", [])
        ingestion_config = config.get("ingestion", {})

        for month in months:
            for key in keys:
                csv_path = ingestion_config["paths"]["csv"][key].format(month=month)
                parquet_path = ingestion_config["paths"]["parquet"][key].format(month=month)
                timestamp_columns = ingestion_config["convert_date_columns"].get(key, {}).get("timestamp_columns", [])
                date_columns = ingestion_config["convert_date_columns"].get(key, {}).get("date_columns", [])

                df = read_csv_with_schema(spark, csv_path, getattr(CustomSchema, key))
                df = convert_date_columns(df, timestamp_columns=timestamp_columns, date_columns=date_columns)
                
                logger.info(f"Writing {key} data to Parquet: {parquet_path}")
                df.write.parquet(parquet_path, mode="overwrite")
                logger.info(f"Successfully wrote {key} data for month {month}")

    except Exception as e:
        logger.error(f"Error during data ingestion: {e}")
        raise
    finally:
        if spark:
            spark.stop()
            logger.info("Spark session stopped")

def main():
    try:
        config = load_config("config.json")
        ingest_data(config)
        logger.info("Data ingestion process completed successfully.")
    except Exception as e:
        logger.error(f"Failed to complete data ingestion: {e}")

if __name__ == "__main__":
    main()