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
    df_eval: DataFrame,
    df_test: DataFrame,
    cols: list,
    scaler,
    input_col: str,
    output_col: str,
) -> tuple:
    """
    Apply scaling (MinMax or Standard) to specified columns in train, eval, and test DataFrames.

    Args:
        df_train (DataFrame): Training DataFrame.
        df_eval (DataFrame): Evaluation DataFrame.
        df_test (DataFrame): Testing DataFrame.
        cols (list): List of column names to scale.
        scaler: Scaler instance (MinMaxScaler or StandardScaler).
        input_col (str): Input column for the scaler.
        output_col (str): Output column for the scaled data.

    Returns:
        tuple: Scaled train, eval, and test DataFrames.
    """
    assembler = VectorAssembler(inputCols=cols, outputCol=input_col)
    df_train = assembler.transform(df_train)
    df_eval = assembler.transform(df_eval)
    df_test = assembler.transform(df_test)

    scaler_model = scaler.setInputCol(input_col).setOutputCol(output_col).fit(df_train)

    df_train_scaled = scaler_model.transform(df_train).drop(input_col)
    df_eval_scaled = scaler_model.transform(df_eval).drop(input_col)
    df_test_scaled = scaler_model.transform(df_test).drop(input_col)

    return df_train_scaled, df_eval_scaled, df_test_scaled

def apply_onehot_encoding(df_train: DataFrame, df_eval: DataFrame, df_test: DataFrame, cols: list) -> tuple:
    """
    Apply StringIndexer and OneHotEncoder to specified categorical columns on train, eval, and test DataFrames.
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
    df_eval_encoded = model.transform(df_eval)
    df_test_encoded = model.transform(df_test)

    return df_train_encoded, df_eval_encoded, df_test_encoded, model

def fill_numerical_with_median(
    df_train: DataFrame, df_eval: DataFrame, df_test: DataFrame, cols: list
) -> tuple:
    """
    Fill null values in specified numerical columns with the median value for train data,
    and apply the same median to the eval and test data.
    """
    for column in cols:
        median_value = df_train.approxQuantile(column, [0.5], 0.01)[0]
        df_train = df_train.fillna({column: median_value})
        df_eval = df_eval.fillna({column: median_value})
        df_test = df_test.fillna({column: median_value})

    return df_train, df_eval, df_test

def fill_categorical_with_mode(
    df_train: DataFrame, df_eval: DataFrame, df_test: DataFrame, cols: list
) -> tuple:
    """
    Fill null values in specified categorical columns with the mode for train data,
    and apply the same mode to the eval and test data.
    """
    for column in cols:
        mode_row = df_train.groupBy(column).count().orderBy("count", ascending=False).first()
        if mode_row:
            mode_value = mode_row[0]
            df_train = df_train.fillna({column: mode_value})
            df_eval = df_eval.fillna({column: mode_value})
            df_test = df_test.fillna({column: mode_value})
        else:
            logger.warning(f"No mode found for column: {column}. Skipping fillna.")

    return df_train, df_eval, df_test

def preprocess(df_train: DataFrame, df_eval: DataFrame, df_test: DataFrame) -> tuple:
    """
    Preprocess the train, eval, and test datasets by applying transformations on numeric and categorical columns.
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
    df_eval = apply_log1p(df_eval, numeric_cols)
    df_test = apply_log1p(df_test, numeric_cols)
    logger.info("Applied log1p to numeric columns")

    # Apply MinMax scaling to numeric columns
    df_train, df_eval, df_test = apply_scaling(
        df_train=df_train,
        df_eval=df_eval,
        df_test=df_test,
        cols=numeric_cols,
        scaler=MinMaxScaler(),
        input_col="features_numeric",
        output_col="scaled_numeric_features",
    )
    logger.info("Applied MinMax scaling to numeric columns")

    # Process age column: Fill nulls, cap, apply log1p, and then Standard scaling
    df_train, df_eval, df_test = fill_numerical_with_median(df_train, df_eval, df_test, [age_col])
    df_train = cap_age(df_train, age_col)
    df_eval = cap_age(df_eval, age_col)
    df_test = cap_age(df_test, age_col)
    df_train = apply_log1p(df_train, [age_col])
    df_eval = apply_log1p(df_eval, [age_col])
    df_test = apply_log1p(df_test, [age_col])

    df_train, df_eval, df_test = apply_scaling(
        df_train=df_train,
        df_eval=df_eval,
        df_test=df_test,
        cols=[age_col],
        scaler=StandardScaler(),
        input_col="features_age",
        output_col="scaled_age",
    )
    logger.info("Processed age column: Filled nulls, capped, applied log1p, and Standard scaling")

    # Process account_balance column: Fill nulls, apply log1p, and Standard scaling
    df_train, df_eval, df_test = fill_numerical_with_median(df_train, df_eval, df_test, [account_balance_col])
    df_train = apply_log1p(df_train, [account_balance_col])
    df_eval = apply_log1p(df_eval, [account_balance_col])
    df_test = apply_log1p(df_test, [account_balance_col])

    df_train, df_eval, df_test = apply_scaling(
        df_train=df_train,
        df_eval=df_eval,
        df_test=df_test,
        cols=[account_balance_col],
        scaler=StandardScaler(),
        input_col="features_account_balance",
        output_col="scaled_account_balance",
    )
    logger.info("Processed account_balance column: Filled nulls, applied log1p, and Standard scaling")

    # Process registration_year column: Fill nulls and apply MinMax scaling
    df_train, df_eval, df_test = fill_categorical_with_mode(
        df_train, df_eval, df_test, [registration_year_col]
    )
    df_train, df_eval, df_test = apply_scaling(
        df_train=df_train,
        df_eval=df_eval,
        df_test=df_test,
        cols=[registration_year_col],
        scaler=MinMaxScaler(),
        input_col="features_registration_year",
        output_col="scaled_registration_year",
    )
    logger.info("Processed registration_year column: Filled nulls and applied MinMax scaling")

    # One-hot encoding for categorical features
    df_train, df_eval, df_test, _ = apply_onehot_encoding(df_train, df_eval, df_test, categorical_cols)
    logger.info("One-hot encoded categorical columns")

    return df_train, df_eval, df_test

def process_and_select_features(df_train: DataFrame, df_eval: DataFrame, df_test: DataFrame) -> tuple:
    """
    Select the final set of features from the processed train, eval, and test DataFrames.

    Args:
        df_train (DataFrame): Preprocessed training DataFrame.
        df_eval (DataFrame): Preprocessed evaluation DataFrame.
        df_test (DataFrame): Preprocessed testing DataFrame.

    Returns:
        tuple: DataFrames with selected features and label columns for train, eval, and test datasets.
    """
    # List of final selected features
    selected_features = [
        "scaled_numeric_features",
        "scaled_age",
        "scaled_account_balance",
        "scaled_registration_year",
        "contract_type_v_encoded",
        "gender_v_encoded",
        "ability_status_encoded",
    ]

    # Assemble selected features into a single vector column "features"
    assembler = VectorAssembler(inputCols=selected_features, outputCol="FEATURES")

    df_train = assembler.transform(df_train)
    df_eval = assembler.transform(df_eval)
    df_test = assembler.transform(df_test)

    # Select only the 'features' and 'label' columns for the final output
    selected_cols = ["bib_id", "FEATURES", "label"]
    df_train = df_train.select(selected_cols)
    df_eval = df_eval.select(selected_cols)
    df_test = df_test.select(selected_cols)

    logger.info("Selected features and label columns for train, eval, and test datasets")

    return df_train, df_eval, df_test

def main():
    """
    Main function for the preprocessing pipeline.
    - Reads train, eval, and test datasets.
    - Applies preprocessing steps (scaling, encoding, and transformations).
    - Saves the processed datasets for model training, evaluation, and testing.
    """
    try:
        # Load configuration
        config = load_config()
        read_path_template = get_config_value(config, "dataset", "path")
        write_path_template = get_config_value(config, "preprocessed", "path")

        # Create Spark session
        spark = create_spark_session("Preprocessing")

        # Read the datasets from Parquet files
        df_train = read_parquet(spark, read_path_template.format(type="train"))
        df_eval = read_parquet(spark, read_path_template.format(type="eval"))
        df_test = read_parquet(spark, read_path_template.format(type="test"))

        logger.info("Successfully read train, eval, and test datasets")

        # Preprocess the datasets
        df_train, df_eval, df_test = preprocess(df_train, df_eval, df_test)

        # Select features and labels
        df_train, df_eval, df_test = process_and_select_features(df_train, df_eval, df_test)

        # Save the processed datasets to Parquet files
        write_parquet(df_train, write_path_template.format(type="train"))
        write_parquet(df_eval, write_path_template.format(type="eval"))
        write_parquet(df_test, write_path_template.format(type="test"))

        logger.info("Processed train, eval, and test datasets saved successfully")

    except Exception as e:
        logger.error(f"Error in preprocessing pipeline: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()