import sys
sys.path.append('/home/sajjad/Projects/ChurnApp')

import os
import json
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, max as spark_max, min as spark_min, datediff, when, greatest, least
from src.logger import setup_logger
from src.utils import load_config, get_config_value, create_spark_session, read_parquet, write_parquet

# Initialize logger
logger = setup_logger(__name__)

import os

def set_label_paths(config: dict) -> None:
    """
    Sets the train and test label paths in the config dictionary based on the months provided in the config.
    
    This function modifies the config in place by updating the 'train' and 'test' paths under the 'label' key.
    
    :param config: Configuration dictionary containing paths and months.
    :raises KeyError: If any required key is missing in the config.
    """
    try:
        # Retrieve base path and month information from config
        base_path = config["label"]["paths"]["base_path"]
        months = config["months"]

        # Create train and test paths
        train_path = os.path.join(base_path, f"train_{months[0]}_{months[1]}")
        test_path = os.path.join(base_path, f"test_{months[1]}_{months[2]}")

        # Update the config with train and test paths
        config["label"]["paths"]["train"] = train_path
        config["label"]["paths"]["test"] = test_path

        # Optionally, save back to file if persistence is required
        with open('config.json', 'w') as f:
            json.dump(config, f, indent=4)

        logger.info(f"Train and test label paths set in config: {train_path}, {test_path}")
        return train_path, test_path

    except KeyError as e:
        logger.error(f"Error in setting label paths: Missing key in configuration: {e}")
        raise

def get_required_paths(
        config: dict, 
        current_month: str, 
        next_month: str) -> dict:
    """
    Retrieves the required file paths for the current and next months from the configuration.
    
    This function dynamically fetches paths for 'package' and 'recharge' parquet files 
    for the current and next months by leveraging the get_config_value function for safe access.

    :param config: Configuration dictionary containing file paths.
    :param current_month: The current month for which paths are required (e.g., '202309').
    :param next_month: The subsequent month for which paths are required (e.g., '202310').
    :return: A dictionary containing the paths for the current and next months.
    :raises KeyError: If any required key is missing in the configuration.
    """
    try:
        # Retrieve paths using get_config_value to ensure keys exist and handle errors
        package_path_template = get_config_value(config, "ingestion", "paths", "parquet", "package")
        recharge_path_template = get_config_value(config, "ingestion", "paths", "parquet", "recharge")

        # Format the paths for the current and next months
        paths = {
            "current_package_path": package_path_template.format(month=current_month),
            "current_recharge_path": recharge_path_template.format(month=current_month),
            "next_package_path": package_path_template.format(month=next_month),
            "next_recharge_path": recharge_path_template.format(month=next_month),
        }

        logger.info(f"Successfully retrieved paths for current month: {current_month} and next month: {next_month}")
        return paths
    
    except KeyError as e:
        logger.error(f"Error retrieving paths from config: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in get_required_paths: {e}")
        raise

def read_required_data(
    spark: SparkSession, 
    config: dict, 
    current_month: str, 
    next_month: str
) -> tuple[DataFrame, DataFrame, DataFrame, DataFrame]:
    """
    Reads the required parquet files for the current and next month.

    :param spark: SparkSession object.
    :param config: Configuration dictionary containing file paths or other relevant parameters.
    :param current_month: Identifier for the current month data.
    :param next_month: Identifier for the next month data.
    :return: A tuple containing DataFrames for current and next month package and recharge data.
    """
    paths = get_required_paths(config, current_month, next_month)
    current_package = read_parquet(spark=spark, parquet_path=paths["current_package_path"])
    current_recharge = read_parquet(spark=spark, parquet_path=paths["current_recharge_path"])
    next_package = read_parquet(spark=spark, parquet_path=paths["next_package_path"])
    next_recharge = read_parquet(spark=spark, parquet_path=paths["next_recharge_path"])
    
    return current_package, current_recharge, next_package, next_recharge

def find_time_activity_of_user(
    package_df: DataFrame, 
    recharge_df: DataFrame, 
    aggregation_func_str: str
) -> DataFrame:
    """
    Computes aggregated activity dates (maximum or minimum) for users based on package and recharge data.

    :param package_df: DataFrame containing package data with 'bib_id' and 'activation_date' columns.
    :param recharge_df: DataFrame containing recharge data with 'bib_id' and 'recharge_dt' columns.
    :param aggregation_func_str: Aggregation function to apply; should be either 'max' or 'min'.
    :return: DataFrame with 'bib_id' and the aggregated date as either 'max_date' or 'min_date'.
    :raises ValueError: If the provided aggregation function is not 'max' or 'min'.
    """

    # Define the aggregation function mapping
    func_mapping = {"max": spark_max, "min": spark_min}
    
    if aggregation_func_str not in func_mapping:
        raise ValueError("Invalid aggregation function. Expected 'max' or 'min'.")
    
    aggregate_func = func_mapping[aggregation_func_str]
    combine_func = greatest if aggregation_func_str == "max" else least

    # Aggregate dates for packages and recharges
    package_agg = package_df.groupBy("bib_id").agg(
        aggregate_func(col("activation_date")).alias("package_date")
    )
    recharge_agg = recharge_df.groupBy("bib_id").agg(
        aggregate_func(col("recharge_dt")).alias("recharge_date")
    )

    # Join aggregated results
    combined_df = package_agg.join(recharge_agg, on="bib_id", how="outer")

    # Compute the final aggregated date
    combined_df = combined_df.withColumn(
        f"{aggregation_func_str}_date",
        combine_func(col("package_date"), col("recharge_date"))
    )
    
    return combined_df.select("bib_id", f"{aggregation_func_str}_date")

def prepare_churn_label(
    current_package_df: DataFrame, 
    current_recharge_df: DataFrame, 
    next_package_df: DataFrame, 
    next_recharge_df: DataFrame
) -> DataFrame:
    """
    Determines churn labels by comparing the latest and earliest activity dates across periods.

    :param current_package_df: DataFrame with package data for the current period.
    :param current_recharge_df: DataFrame with recharge data for the current period.
    :param next_package_df: DataFrame with package data for the next period.
    :param next_recharge_df: DataFrame with recharge data for the next period.
    :return: DataFrame with 'bib_id' and a churn label (1 for churned, 0 for not churned).
    """
    # Compute maximum dates for the current period
    latest_activity_current = find_time_activity_of_user(current_package_df, current_recharge_df, aggregation_func_str="max")
    
    # Compute minimum dates for the next period
    earliest_activity_next = find_time_activity_of_user(next_package_df, next_recharge_df, aggregation_func_str="min")

    # Join the DataFrames on 'bib_id'
    df = latest_activity_current.join(earliest_activity_next, on="bib_id", how="left")

    # Calculate date difference and determine churn label
    df = df.withColumn(
        "date_diff",
        datediff(col("min_date"), col("max_date"))
    ).withColumn(
        "label",
        when(
            (col("min_date").isNull() & col("max_date").isNotNull()) | (col("date_diff") > 30),
            1
        ).otherwise(0)
    )

    return df.select("bib_id", "label")

def generate_churn_label(
    spark: SparkSession, 
    config: dict, 
    current_month: str, 
    next_month: str
) -> DataFrame:
    """
    Generates churn labels by reading data for the current and next month and determining churn status.

    :param spark: SparkSession object.
    :param config: Configuration dictionary containing file paths or other relevant parameters.
    :param current_month: Identifier for the current month data.
    :param next_month: Identifier for the next month data.
    :return: DataFrame with 'bib_id' and churn labels.
    """
    return prepare_churn_label(*read_required_data(spark, config, current_month, next_month))

def main():
    spark = None
    try:
        # Load configuration
        config = load_config("config.json")
        train_label_path, test_label_path = set_label_paths(config)

        # Create Spark session
        spark = create_spark_session("PrepareLabel")
        spark.conf.set("spark.sql.parquet.datetimeRebaseModeInWrite", "CORRECTED")

        # Generate train and test labels
        months = get_config_value(config, "months")
        train_label_df = generate_churn_label(spark, config, months[0], months[1])
        test_label_df = generate_churn_label(spark, config, months[1], months[2])

        # Write the output Parquet files
        write_parquet(train_label_df, train_label_path)
        write_parquet(test_label_df, test_label_path)

        logger.info("Label preparation completed successfully.")

    except FileNotFoundError as fnf_error:
        logger.error(f"Config file not found: {fnf_error}")
    except KeyError as key_error:
        logger.error(f"Missing key in configuration: {key_error}")
    except Exception as e:
        logger.error(f"Failed to complete label preparation: {e}")
        raise
    finally:
        if spark:
            spark.stop()
            logger.info("Spark session stopped.")

if __name__ == "__main__":
    main()
    
