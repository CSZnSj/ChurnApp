# preprocess.py

import sys
sys.path.append('/home/sajjad/Projects/ChurnApp')

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import log1p, when, col
from pyspark.ml.feature import MinMaxScaler, StandardScaler, OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml import Pipeline
from src.logger import setup_logger
from src.utils import load_config, get_config_value, create_spark_session, read_parquet, write_parquet

logger = setup_logger(__name__)

def apply_log1p(df: DataFrame, cols: list) -> DataFrame:
    """
    Apply log1p (log(x + 1)) transformation to specified numeric columns.
    """
    for col_name in cols:
        df = df.withColumn(col_name, log1p(col(col_name)))
    return df

def cap_age(df: DataFrame, age_col: str) -> DataFrame:
    """
    Cap age values at 100 (if age > 100, set it to 100).
    """
    return df.withColumn(age_col, when(col(age_col) > 100, 100).otherwise(col(age_col)))


def apply_scaling(
    df_train: DataFrame,
    df_test: DataFrame,
    cols: list,
    scaler,
    input_col: str,
    output_col: str,
) -> tuple:
    """
    Apply scaling (MinMax or Standard) to specified columns in both train and test DataFrames.

    Args:
        df_train (DataFrame): Training DataFrame.
        df_test (DataFrame): Testing DataFrame.
        cols (list): List of column names to scale.
        scaler: Scaler instance (MinMaxScaler or StandardScaler).
        input_col (str): Input column for the scaler.
        output_col (str): Output column for the scaled data.

    Returns:
        tuple: Scaled train and test DataFrames.
    """
    assembler = VectorAssembler(inputCols=cols, outputCol=input_col)
    df_train = assembler.transform(df_train)
    df_test = assembler.transform(df_test)

    scaler_model = scaler.setInputCol(input_col).setOutputCol(output_col).fit(df_train)

    df_train_scaled = scaler_model.transform(df_train).drop(input_col)
    df_test_scaled = scaler_model.transform(df_test).drop(input_col)

    return df_train_scaled, df_test_scaled


def apply_onehot_encoding(df_train: DataFrame, df_test: DataFrame, cols: list) -> tuple:
    """
    Apply StringIndexer and OneHotEncoder to specified categorical columns on train and test DataFrames.
    """
    indexers = [
        StringIndexer(inputCol=col, outputCol=f"{col}_indexed", handleInvalid="keep")
        for col in cols
    ]
    encoders = [
        OneHotEncoder(inputCol=f"{col}_indexed", outputCol=f"{col}_encoded")
        for col in cols
    ]
    pipeline = Pipeline(stages=indexers + encoders)

    model = pipeline.fit(df_train)
    df_train_encoded = model.transform(df_train)
    df_test_encoded = model.transform(df_test)

    return df_train_encoded, df_test_encoded, model


def fill_numerical_with_median(
    df_train: DataFrame, df_test: DataFrame, cols: list
) -> tuple:
    """
    Fill null values in specified numerical columns with the median value for train data,
    and apply the same median to the test data.
    """
    for column in cols:
        median_value = df_train.approxQuantile(column, [0.5], 0.01)[0]
        df_train = df_train.fillna({column: median_value})
        df_test = df_test.fillna({column: median_value})

    return df_train, df_test


def fill_categorical_with_mode(
    df_train: DataFrame, df_test: DataFrame, cols: list
) -> tuple:
    """
    Fill null values in specified categorical columns with the mode for train data,
    and apply the same mode to the test data.
    """
    for column in cols:
        mode_row = df_train.groupBy(column).count().orderBy("count", ascending=False).first()
        if mode_row:
            mode_value = mode_row[0]
            df_train = df_train.fillna({column: mode_value})
            df_test = df_test.fillna({column: mode_value})
        else:
            logger.warning(f"No mode found for column: {column}. Skipping fillna.")

    return df_train, df_test

def preprocess(df_train: DataFrame, df_test: DataFrame) -> tuple:
    """
    Preprocess the train and test datasets by applying transformations on numeric and categorical columns.
    """
    # Define columns for processing
    numeric_cols = [
        "CountAssignLoanAmount",
        "MedianAssignLoanAmount",
        "CountRecoveryLoanAmount",
        "MedianRecoveryLoanAmount",
        "CountPackage",
        "MedianPackageAmount",
        "MedianPackageDuration",
        "CountRecharge",
        "MedianRechargeAmount",
    ]
    age_col = "age"
    account_balance_col = "account_balance"
    registration_year_col = "registration_year"
    categorical_cols = ["contract_type_v", "gender_v", "ability_status"]

    # Apply log1p to numeric columns
    df_train = apply_log1p(df_train, numeric_cols)
    df_test = apply_log1p(df_test, numeric_cols)
    logger.info("Applied log1p to numeric columns")

    # Apply MinMax scaling to numeric columns
    df_train, df_test = apply_scaling(
        df_train=df_train,
        df_test=df_test,
        cols=numeric_cols,
        scaler=MinMaxScaler(),
        input_col="features_numeric",
        output_col="scaled_numeric_features",
    )
    logger.info("Applied MinMax scaling to numeric columns")

    # Process age column: Fill nulls, cap, apply log1p, and then Standard scaling
    df_train, df_test = fill_numerical_with_median(df_train, df_test, [age_col])
    df_train = cap_age(df_train, age_col)
    df_test = cap_age(df_test, age_col)
    df_train = apply_log1p(df_train, [age_col])
    df_test = apply_log1p(df_test, [age_col])

    df_train, df_test = apply_scaling(
        df_train=df_train,
        df_test=df_test,
        cols=[age_col],
        scaler=StandardScaler(),
        input_col="features_age",
        output_col="scaled_age",
    )
    logger.info("Processed age column: Filled nulls, capped, applied log1p, and Standard scaling")

    # Process account_balance column: Fill nulls, apply log1p, and Standard scaling
    df_train, df_test = fill_numerical_with_median(df_train, df_test, [account_balance_col])
    df_train = apply_log1p(df_train, [account_balance_col])
    df_test = apply_log1p(df_test, [account_balance_col])

    df_train, df_test = apply_scaling(
        df_train=df_train,
        df_test=df_test,
        cols=[account_balance_col],
        scaler=StandardScaler(),
        input_col="features_account_balance",
        output_col="scaled_account_balance",
    )
    logger.info("Processed account_balance column: Filled nulls, applied log1p, and Standard scaling")

    # Process registration_year column: Fill nulls and apply MinMax scaling
    df_train, df_test = fill_categorical_with_mode(
        df_train, df_test, [registration_year_col]
    )
    df_train, df_test = apply_scaling(
        df_train=df_train,
        df_test=df_test,
        cols=[registration_year_col],
        scaler=MinMaxScaler(),
        input_col="features_registration_year",
        output_col="scaled_registration_year",
    )
    logger.info("Processed registration_year column: Filled nulls and applied MinMax scaling")

    # One-hot encoding for categorical features
    df_train, df_test, _ = apply_onehot_encoding(df_train, df_test, categorical_cols)
    logger.info("One-hot encoded categorical columns")

    return df_train, df_test

def process_and_select_features(df_train: DataFrame, df_test: DataFrame) -> tuple:
    """
    Transform and select relevant features from both train and test DataFrames.
    """
    # Define the columns that need to be assembled
    scaled_numeric = "scaled_numeric_features"
    scaled_age = "scaled_age"
    scaled_account_balance = "scaled_account_balance"
    scaled_registration_year = "scaled_registration_year"
    encoded_contract_type = "contract_type_v_encoded"
    encoded_gender = "gender_v_encoded"
    encoded_ability_status = "ability_status_encoded"

    index_cols = [
        scaled_numeric,
        scaled_age,
        scaled_account_balance,
        scaled_registration_year,
        encoded_contract_type,
        encoded_gender,
        encoded_ability_status,
    ]

    # Assemble all features into a single vector
    assembler = VectorAssembler(inputCols=index_cols, outputCol="FEATURES")
    df_train = assembler.transform(df_train)
    df_test = assembler.transform(df_test)

    # Select the relevant columns for final output
    select_cols = ["bib_id", "FEATURES", "label"]
    df_train_selected = df_train.select(select_cols)
    df_test_selected = df_test.select(select_cols)

    return df_train_selected, df_test_selected

def main():
    """
    Main function to load raw train and test data, apply preprocessing, and save the processed data.
    """
    try:
        config = load_config("config.json")
        read_path_template = get_config_value(config, "dataset", "path")
        write_path_template = get_config_value(config, "preprocessed", "path")

        spark = create_spark_session("Preprocess")
        spark.conf.set("spark.sql.parquet.datetimeRebaseModeInWrite", "CORRECTED")

        train_df = read_parquet(spark, read_path_template.format(type="train"))
        test_df = read_parquet(spark, read_path_template.format(type="test"))

        train_df, test_df = preprocess(train_df, test_df)
        train_df, test_df = process_and_select_features(train_df, test_df)

        write_parquet(train_df, write_path_template.format(type="train"))
        write_parquet(test_df, write_path_template.format(type="test"))

        logger.info("Preprocessing completed successfully")

    except FileNotFoundError as fnf_error:
        logger.error(f"Configuration file not found: {fnf_error}")
        raise
    except Exception as e:
        logger.error(f"Error during preprocessing: {str(e)}")
        raise
    finally:
        if spark:
            spark.stop()
            logger.info("Spark session stopped.")

if __name__ == "__main__":
    main()
