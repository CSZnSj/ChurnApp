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

def ingest_data(config: dict) -> None:
    """
    Orchestrates the data ingestion process from CSV to Parquet format.
    """
    logger.info("Starting data ingestion process...")
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
                write_parquet(df, parquet_path, mode = "overwrite")
                logger.info(f"Successfully wrote {key} data for month {month}")

    except Exception as e:
        logger.error(f"Error during data ingestion: {e}")
        raise
    finally:
        spark.stop()
        logger.info("Spark session stopped")

def main():
    try:
        config = load_config("config.json")
        ingest_data(config)
        logger.info("Data ingestion process completed successfully.")
    except Exception as e:
        logger.error(f"Failed to complete data ingestion: {e}")
        raise

if __name__ == "__main__":
    main()