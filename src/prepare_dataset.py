# prepare_feature.py

import sys
sys.path.append('/home/sajjad/Projects/ChurnApp')

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, year, current_date, datediff, median, count
from pyspark.sql.utils import AnalysisException
from typing import Tuple
from src.logger import setup_logger
from src.utils import load_config, get_config_value, create_spark_session, write_parquet, read_parquet, remove_data

logger = setup_logger(__name__)

def read_required_data(
    spark: SparkSession,
    config: dict, 
    month: str
) -> Tuple[DataFrame, DataFrame, DataFrame, DataFrame, DataFrame]:
    """
    Reads the necessary Parquet files into DataFrames for a given month.

    Args:
        spark (SparkSession): An active Spark session object.
        config (dict): Configuration dictionary containing file paths or other relevant parameters.
        month (str): The month string to be used in the file paths.

    Returns:
        Tuple[DataFrame, DataFrame, DataFrame, DataFrame, DataFrame]: 
        DataFrames for loan assignment, loan recovery, package, recharge, and user data.

    Raises:
        ValueError: If an error occurs while reading the Parquet files.
    """
    try:
        base = get_config_value(config, "ingestion", "paths", "parquet")
        assign_df = read_parquet(spark=spark, parquet_path=base["loan_assign"].format(month=month))
        recovery_df = read_parquet(spark=spark, parquet_path=base["loan_recovery"].format(month=month))
        package_df = read_parquet(spark=spark, parquet_path=base["package"].format(month=month))
        recharge_df = read_parquet(spark=spark, parquet_path=base["recharge"].format(month=month))
        user_df = read_parquet(spark=spark, parquet_path=base["user"].format(month=month))
        
        logger.info("Successfully read all required data for month: %s", month)
        return assign_df, recovery_df, package_df, recharge_df, user_df
    
    except Exception as e:
        logger.error("Error reading required data for month %s: %s", month, str(e))
        raise ValueError("Failed to read required data")

def prepare_assign_df(assign_df: DataFrame) -> DataFrame:
    """
    Processes the assignment DataFrame to compute the count and median loan amount per 'bib_id'.

    Args:
        assign_df (DataFrame): DataFrame containing assignment data.

    Returns:
        DataFrame: Processed DataFrame with the count and median loan amount per 'bib_id'.
    
    Raises:
        AnalysisException: If an error occurs during aggregation.
    """
    try:
        aggregated_assign_df = assign_df.groupBy("bib_id").agg(
            count("*").alias("CountAssignLoanAmount"),
            median("loan_amount").alias("MedianAssignLoanAmount"),
        )
        logger.info("Successfully processed assignment DataFrame")
        return aggregated_assign_df
    
    except AnalysisException as e:
        logger.error("Error processing assignment DataFrame: %s", str(e))
        raise

def prepare_recovery_df(recovery_df: DataFrame) -> DataFrame:
    """
    Processes the recovery DataFrame to compute the count and median recovery loan amount per 'bib_id'.

    Args:
        recovery_df (DataFrame): DataFrame containing recovery data.

    Returns:
        DataFrame: Processed DataFrame with the count and median recovery loan amount per 'bib_id'.

    Raises:
        AnalysisException: If an error occurs during aggregation.
    """
    try:
        aggregated_recovery_df = recovery_df.groupBy("bib_id").agg(
            count("*").alias("CountRecoveryLoanAmount"),
            median("loan_amount").alias("MedianRecoveryLoanAmount"),
        )
        logger.info("Successfully processed recovery DataFrame")
        return aggregated_recovery_df
    
    except AnalysisException as e:
        logger.error("Error processing recovery DataFrame: %s", str(e))
        raise

def prepare_package_df(package_df: DataFrame) -> DataFrame:
    """
    Processes the package DataFrame to compute the count of packages and median package amount/duration per 'bib_id'.
    Calculates the duration based on activation and deactivation dates.

    Args:
        package_df (DataFrame): DataFrame containing package data.

    Returns:
        DataFrame: Processed DataFrame with the count, median package amount, and median duration per 'bib_id'.

    Raises:
        AnalysisException: If an error occurs during aggregation.
    """
    try:
        # Calculate duration between activation and deactivation dates
        package_df = package_df.withColumn("duration", datediff(col("deactivation_date"), col("activation_date")))

        # Group by 'bib_id' and calculate required statistics
        aggregated_package_df = package_df.groupBy("bib_id").agg(
            count("*").alias("CountPackage"),
            median("offer_amount").alias("MedianPackageAmount"),
            median("duration").alias("MedianPackageDuration"),
        )
        logger.info("Successfully processed package DataFrame")
        return aggregated_package_df
    
    except AnalysisException as e:
        logger.error("Error processing package DataFrame: %s", str(e))
        raise

def prepare_recharge_df(recharge_df: DataFrame) -> DataFrame:
    """
    Processes the recharge DataFrame to compute the count and median recharge amount per 'bib_id'.

    Args:
        recharge_df (DataFrame): DataFrame containing recharge data.

    Returns:
        DataFrame: Processed DataFrame with the count and median recharge amount per 'bib_id'.

    Raises:
        AnalysisException: If an error occurs during aggregation.
    """
    try:
        aggregated_recharge_df = recharge_df.groupBy("bib_id").agg(
            count("*").alias("CountRecharge"),
            median("recharge_value_amt").alias("MedianRechargeAmount"),
        )
        logger.info("Successfully processed recharge DataFrame")
        return aggregated_recharge_df
    
    except AnalysisException as e:
        logger.error("Error processing recharge DataFrame: %s", str(e))
        raise

def prepare_user_df(user_df: DataFrame) -> DataFrame:
    """
    Processes the user DataFrame by calculating the age, extracting registration year, and selecting relevant columns.

    Args:
        user_df (DataFrame): DataFrame containing user data.

    Returns:
        DataFrame: Transformed DataFrame with calculated age, registration year, and selected columns.

    Raises:
        AnalysisException: If an error occurs during transformation or column selection.
    """
    try:
        # Calculate age from 'date_of_birth_d'
        user_df = user_df.withColumn("age", datediff(current_date(), col("date_of_birth_d")) / 365.25)

        # Extract registration year from 'registration_date_d'
        user_df = user_df.withColumn("registration_year", year(col("registration_date_d")))

        # Define relevant columns to select
        selected_columns = ["bib_id", "age", "registration_year", "contract_type_v", "gender_v", "ability_status", "account_balance"]
        user_df = user_df.select(selected_columns)
        
        logger.info("Successfully processed user DataFrame")
        return user_df
    
    except AnalysisException as e:
        logger.error("Error processing user DataFrame: %s", str(e))
        raise

def prepare_data(
    assign_df: DataFrame,
    recovery_df: DataFrame,
    package_df: DataFrame,
    recharge_df: DataFrame,
    user_df: DataFrame
) -> Tuple[DataFrame, DataFrame, DataFrame, DataFrame, DataFrame]:
    """
    Prepares all data by applying respective transformation functions to each DataFrame.

    Args:
        assign_df (DataFrame): Assignment data.
        recovery_df (DataFrame): Recovery data.
        package_df (DataFrame): Package data.
        recharge_df (DataFrame): Recharge data.
        user_df (DataFrame): User data.

    Returns:
        Tuple[DataFrame, DataFrame, DataFrame, DataFrame, DataFrame]: Processed DataFrames for assignment, recovery, package, recharge, and user data.
    """
    return (
        prepare_assign_df(assign_df),
        prepare_recovery_df(recovery_df),
        prepare_package_df(package_df),
        prepare_recharge_df(recharge_df),
        prepare_user_df(user_df)
    )

def merge_dataframes(
    assign_df: DataFrame,
    recovery_df: DataFrame,
    package_df: DataFrame,
    recharge_df: DataFrame,
    user_df: DataFrame,
    churn_label_df: DataFrame
) -> DataFrame:
    """
    Merges multiple DataFrames with churn labels based on 'bib_id'.

    Args:
        assign_df (DataFrame): Processed assignment data.
        recovery_df (DataFrame): Processed recovery data.
        package_df (DataFrame): Processed package data.
        recharge_df (DataFrame): Processed recharge data.
        user_df (DataFrame): Processed user data.
        churn_label_df (DataFrame): Churn label data.

    Returns:
        DataFrame: Merged DataFrame containing all input data.
    """
    merged_df = churn_label_df \
        .join(assign_df, on="bib_id", how="left") \
        .join(recovery_df, on="bib_id", how="left") \
        .join(package_df, on="bib_id", how="left") \
        .join(recharge_df, on="bib_id", how="left") \
        .join(user_df, on="bib_id", how="left")

    logger.info("Successfully merged all DataFrames")
    return merged_df

def generate_dataset(
    spark: SparkSession,
    config: dict,
    month: str,
    churn_label_df: DataFrame
) -> DataFrame:
    """
    Generates a dataset by reading required data, preparing it, and merging it with churn labels df.

    Args:
        spark (SparkSession): Active Spark session.
        config (dict): Configuration dictionary containing file paths or other relevant parameters.
        month (str): Month string to locate the specific data.
        churn_label_df (DataFrame): DataFrame containing churn labels.

    Returns:
        DataFrame: Final prepared and merged dataset for modeling.

    Raises:
        Exception: If any of the preparation steps fail.
    """
    try:
        # Step 1: Read required data
        assign_df, recovery_df, package_df, recharge_df, user_df = read_required_data(spark, config, month)
        
        # Step 2: Prepare each dataset
        assign_df, recovery_df, package_df, recharge_df, user_df = prepare_data(assign_df, recovery_df, package_df, recharge_df, user_df)
        
        # Step 3: Merge all DataFrames with churn labels
        merged_df = merge_dataframes(assign_df, recovery_df, package_df, recharge_df, user_df, churn_label_df)
        
        logger.info("Dataset generated successfully for month: %s", month)
        return merged_df

    except Exception as e:
        logger.error("Failed to generate dataset for month %s: %s", month, str(e))
        raise

def main():
    """
    Main function to handle the dataset generation process for both train and test data.

    This function:
    1. Loads the configuration from a JSON file.
    2. Retrieves the list of months to process and the paths for reading and writing data.
    3. Creates a Spark session with appropriate settings.
    4. Iterates over the months to process, generating a dataset for each month.
    5. Writes the generated datasets to Parquet files.
    6. Removes the used label files to optimize storage.

    The Spark session is ensured to be stopped after the process, even in case of errors.

    Raises:
        Exception: Any errors encountered during the dataset generation process will be logged and re-raised.
    """
    spark = None
    try:
        # Load configuration
        config = load_config("config.json")
        months = get_config_value(config, "months")
        read_path_template = get_config_value(config, "label", "path")
        write_path_template = get_config_value(config, "dataset", "path")

        # Create Spark session
        spark = create_spark_session("DataIngestion")
        # Set Spark config for writing Parquet with corrected datetime rebase mode
        spark.conf.set("spark.sql.parquet.datetimeRebaseModeInWrite", "CORRECTED")

        # Loop through months and data types (train, test)
        for month, data_type in zip(months[:-1], ["train", "test"]):
            # Format paths using the current month and data type
            read_path = read_path_template.format(type=data_type)
            write_path = write_path_template.format(month=month)

            # Read churn label DataFrame from the Parquet file
            churn_label_df = read_parquet(spark=spark, parquet_path=read_path)

            # Generate the dataset for the given month
            dataset_df = generate_dataset(spark=spark, config=config, month=month, churn_label_df=churn_label_df)

            # Write the generated dataset to a Parquet file
            write_parquet(df=dataset_df, output_parquet_path=write_path)

            # Remove the read label file to optimize storage
            remove_data(data_path=read_path)

        logger.info("Dataset preparation completed successfully.")

    except FileNotFoundError as fnf_error:
        # Handle missing config file
        logger.error(f"Configuration file not found: {fnf_error}")
        raise
    except Exception as e:
        # Log any other exceptions and raise them
        logger.error("Error in dataset generation process: %s", str(e))
        raise

    finally:
        # Ensure the Spark session is stopped, even if an error occurs
        if spark:
            spark.stop()
            logger.info("Spark session stopped.")

if __name__ == "__main__":
    main()


