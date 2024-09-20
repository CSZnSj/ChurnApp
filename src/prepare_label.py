import sys
sys.path.append('/home/sajjad/Projects/ChurnApp')

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, max as spark_max, min as spark_min, datediff, when, greatest, least
from src.logger import setup_logger
from src.utils import load_config, get_config_value, create_spark_session, read_parquet, write_parquet

# Initialize logger
logger = setup_logger(__name__)

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
    """
    Main function for preparing churn labels for train and test datasets.

    This function:
    1. Loads configuration from a JSON file.
    2. Retrieves the list of months and base path for label files.
    3. Creates a Spark session with appropriate settings.
    4. Generates churn labels for both train and test datasets based on sequential months.
    5. Writes the generated labels to Parquet files.

    The Spark session is ensured to be stopped after the process, even in case of errors.

    Raises:
        FileNotFoundError: If the configuration file is not found.
        Exception: Any other exceptions encountered during the process will be logged and re-raised.
    """
    spark = None
    try:
        # Load configuration
        config = load_config("config.json")
        months = get_config_value(config, "months")
        base_path = get_config_value(config, "label", "path")

        # Create Spark session
        spark = create_spark_session("PrepareLabel")
        # Set Spark config for writing Parquet with corrected datetime rebase mode
        spark.conf.set("spark.sql.parquet.datetimeRebaseModeInWrite", "CORRECTED")

        # Generate churn labels for train (first two months) and eval (second and third months)
        train_label_df = generate_churn_label(spark=spark, config=config, current_month=months[0], next_month=months[1])
        eval_label_df = generate_churn_label(spark=spark, config=config, current_month=months[1], next_month=months[2])

        # Write the generated churn labels to Parquet files
        write_parquet(df=train_label_df, output_parquet_path=base_path.format(type="train"))
        write_parquet(df=eval_label_df, output_parquet_path=base_path.format(type="eval"))

        logger.info("Label preparation completed successfully.")

    except FileNotFoundError as fnf_error:
        # Handle missing config file
        logger.error(f"Configuration file not found: {fnf_error}")
        raise
    except Exception as e:
        # Log any other exceptions and raise them
        logger.error(f"Failed to complete label preparation: {e}")
        raise

    finally:
        # Ensure the Spark session is stopped, even in case of error
        if spark:
            spark.stop()
            logger.info("Spark session stopped.")


if __name__ == "__main__":
    main()
    
