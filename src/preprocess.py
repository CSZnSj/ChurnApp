# preprocess.py

import sys
sys.path.append('/home/sajjad/Projects/ChurnApp')

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import log1p, when, col, desc
from pyspark.ml.feature import MinMaxScaler, StandardScaler, OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml import Pipeline
from src.logger import setup_logger
from src.utils import load_config, get_config_value, create_spark_session, read_parquet, write_parquet

logger = setup_logger(__name__)

def apply_log1p(df: DataFrame, cols: list) -> DataFrame:
    """
    Apply log1p (log(x + 1)) transformation to specified numeric columns.
    
    Args:
        df (DataFrame): Input DataFrame.
        cols (list): List of column names to apply log1p transformation.
    
    Returns:
        DataFrame: DataFrame with log1p applied to the specified columns.
    """
    for col_name in cols:
        df = df.withColumn(col_name, log1p(col(col_name)))
    return df

def cap_age(df: DataFrame, age_col: str) -> DataFrame:
    """
    Cap age values at 100 (if age > 100, set it to 100) and return the modified DataFrame.
    
    Args:
        df (DataFrame): Input DataFrame.
        age_col (str): Name of the age column to cap values.
    
    Returns:
        DataFrame: DataFrame with capped age values.
    """
    df = df.withColumn(age_col, when(col(age_col) > 100, 100).otherwise(col(age_col)))
    return df

from pyspark.ml.feature import VectorAssembler, MinMaxScaler, StandardScaler
from pyspark.sql import DataFrame

def apply_scaling(df_train: DataFrame, df_test: DataFrame, cols: list, scaler, output_col: str = "features"):
    """
    Apply scaling (MinMax or Standard) to the specified columns on both train and test DataFrames.
    Fit the scaler on the train DataFrame and use the fitted model to transform both the train and test DataFrames.
    
    Args:
        df_train (DataFrame): Training DataFrame.
        df_test (DataFrame): Testing DataFrame.
        cols (list): List of column names to scale.
        scaler: Scaler model to use (MinMaxScaler or StandardScaler).
        output_col (str): The name of the output column for assembled features.
    
    Returns:
        df_train_scaled (DataFrame): Scaled training DataFrame.
        df_test_scaled (DataFrame): Scaled testing DataFrame.
        scaler_model: Fitted scaler model (can be reused).
    """
    
    # VectorAssembler for the columns to scale
    assembler = VectorAssembler(inputCols=cols, outputCol=output_col)
    
    # Assemble features for both train and test datasets
    df_train = assembler.transform(df_train)
    df_test = assembler.transform(df_test)
    
    # Fit the scaler on the training data
    scaler_model = scaler.fit(df_train)
    
    # Apply scaling (fit_transform on train, transform on test)
    df_train_scaled = scaler_model.transform(df_train)
    df_test_scaled = scaler_model.transform(df_test)
    
    # Drop the assembled features column after scaling (optional)
    df_train_scaled = df_train_scaled.drop(output_col)
    df_test_scaled = df_test_scaled.drop(output_col)
    
    return df_train_scaled, df_test_scaled, scaler_model

def apply_onehot_encoding(df_train: DataFrame, df_test: DataFrame, cols: list):
    """
    Apply StringIndexer and OneHotEncoder to specified categorical columns on both train and test DataFrames.
    Fit the encoders on the train DataFrame and use the fitted model to transform both train and test DataFrames.
    
    Args:
        df_train (DataFrame): Training DataFrame.
        df_test (DataFrame): Testing DataFrame.
        cols (list): List of column names to encode.
    
    Returns:
        df_train_encoded (DataFrame): Training DataFrame with one-hot encoded features.
        df_test_encoded (DataFrame): Testing DataFrame with one-hot encoded features.
        model: Fitted pipeline model.
    """
    
    # First apply StringIndexer to convert strings to numerical indices
    indexers = [StringIndexer(inputCol=col, outputCol=f"{col}_indexed", handleInvalid="keep") for col in cols]
    
    # Then apply OneHotEncoder on the indexed columns
    encoders = [OneHotEncoder(inputCol=f"{col}_indexed", outputCol=f"{col}_encoded") for col in cols]
    
    # Create a pipeline with both the indexers and encoders
    pipeline = Pipeline(stages=indexers + encoders)
    
    # Fit the pipeline on the training data
    model = pipeline.fit(df_train)
    
    # Apply the transformation on both train and test datasets
    df_train_encoded = model.transform(df_train)
    df_test_encoded = model.transform(df_test)
    
    return df_train_encoded, df_test_encoded, model

def fill_numerical_with_median(df_train: DataFrame, df_test: DataFrame, cols: list) -> tuple[DataFrame, DataFrame]:
    """
    Fill null values in the specified numerical columns with the median value for train data
    and apply the same transformation to the test data.
    
    Parameters:
    df_train (DataFrame): Training dataset.
    df_test (DataFrame): Testing dataset.
    column (str): The column to apply the transformation on. Default is 'age'.

    Returns:
    (DataFrame, DataFrame): Tuple of transformed train and test DataFrames.
    """
    for column in cols:
        # Calculate the median for the 'age' column in the training dataset
        median_age = df_train.approxQuantile(column, [0.5], 0.01)[0]
        
        # Fill null values with the median in the train DataFrame
        df_train = df_train.fillna({column: median_age})
        
        # Apply the same median to fill null values in the test DataFrame
        df_test = df_test.fillna({column: median_age})
    
    return df_train, df_test

def fill_categorical_with_mode(df_train: DataFrame, df_test: DataFrame, cols: list) -> tuple[DataFrame, DataFrame]:
    """
    Fill null values in the specified categorical column with the most frequent category value (mode)
    for train data and apply the same transformation to the test data.

    Parameters:
    df_train (DataFrame): Training dataset.
    df_test (DataFrame): Testing dataset.
    column (str): The categorical column to apply the transformation on.

    Returns:
    (DataFrame, DataFrame): Tuple of transformed train and test DataFrames.
    """
    for column in cols:
        # Calculate the mode for the specified categorical column in the training dataset
        mode_value = df_train.agg({column: 'mode'}).first()[0]

        # Fill null values with the mode in the train DataFrame
        df_train = df_train.fillna({column: mode_value})
        
        # Apply the same mode to fill null values in the test DataFrame
        df_test = df_test.fillna({column: mode_value})

    return df_train, df_test

def preprocess(df_train: DataFrame, df_test: DataFrame) -> tuple:
    """
    Preprocess the train and test datasets by applying various transformations:
    - Log1p and scaling on numeric features.
    - Capping and scaling on age.
    - Scaling on account balance and registration year.
    - One-hot encoding on categorical features.
    
    Args:
        df_train (DataFrame): Train dataset.
        df_test (DataFrame): Test dataset.
    
    Returns:
        tuple: Processed train and test DataFrames.
    """
    # Define columns for processing
    numeric_cols = [
        "CountAssignLoanAmount", "MedianAssignLoanAmount", "CountRecoveryLoanAmount", 
        "MedianRecoveryLoanAmount", "CountPackage", "MedianPackageAmount", 
        "MedianPackageDuration", "CountRecharge", "MedianRechargeAmount"
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
    min_max_scaler_numericals = MinMaxScaler(inputCol="features", outputCol="scaled_numeric_features")
    df_train, df_test, _ = apply_scaling(df_train=df_train, df_test=df_test, cols=numeric_cols, scaler=min_max_scaler_numericals)

    logger.info("Applied MinMax scaling to numeric columns")

    # Process age column: Cap at 100, apply log1p, and then Standard scaling
    df_train, df_test = fill_numerical_with_median(df_train, df_test, cols=[age_col])
    df_train = cap_age(df_train, age_col)
    df_test = cap_age(df_test, age_col)
    df_train = apply_log1p(df_train, [age_col])
    df_test = apply_log1p(df_test, [age_col])
    
    std_scaler_age = StandardScaler(inputCol="features", outputCol="scaled_age")
    df_train, df_test, _ = apply_scaling(df_train=df_train, df_test=df_test, cols=[age_col], scaler=std_scaler_age)

    logger.info("Processed age column: Cap at 100, apply log1p, and then Standard scaling")

    # Process account_balance column: Apply log1p and Standard scaling
    df_train, df_test = fill_numerical_with_median(df_train, df_test, cols=[account_balance_col])
    df_train = apply_log1p(df_train, [account_balance_col])
    df_test = apply_log1p(df_test, [account_balance_col])

    std_scaler_account_balance = StandardScaler(inputCol="features", outputCol="scaled_account_balance")
    df_train, df_test, _ = apply_scaling(df_train=df_train, df_test=df_test, cols=[account_balance_col], scaler=std_scaler_account_balance)

    logger.info("Processed account_balance column: Apply log1p and Standard scaling")

    # Process registration_year column: Apply MinMax scaling
    df_train, df_test = fill_categorical_with_mode(df_train, df_test, cols=[registration_year_col])
    min_max_scaler_year = MinMaxScaler(inputCol="features", outputCol="scaled_registration_year")
    df_train, df_test, _ = apply_scaling(df_train=df_train, df_test=df_test, cols=[registration_year_col], scaler=min_max_scaler_year)

    logger.info("Processed registration_year column: Apply MinMax scaling")

    # One-hot encoding for categorical features
    df_train, df_test = fill_categorical_with_mode(df_train=df_train, df_test=df_test, cols=categorical_cols)
    df_train, df_test, _ = apply_onehot_encoding(df_train=df_train, df_test=df_test, cols=categorical_cols)
    
    logger.info("One-hot encoded for categorical features")

    return df_train, df_test

def process_and_select_features(df_train: DataFrame, df_test: DataFrame) -> tuple:
    """
    Transforms categorical and numerical features in both train and test DataFrames, 
    and selects the relevant columns for further processing.

    Parameters:
    ----------
    df_train : pyspark.sql.DataFrame
        The training dataset with features and labels.
    df_test : pyspark.sql.DataFrame
        The test dataset with features and labels.

    Returns:
    -------
    tuple (DataFrame, DataFrame)
        Transformed and selected train and test DataFrames with relevant features.
    """
    # Define the columns that need to be encoded and scaled
    index_cols = [
        "scaled_numeric_features", 
        "scaled_age", 
        "scaled_account_balance", 
        "scaled_registration_year", 
        "contract_type_v_encoded", 
        "gender_v_encoded", 
        "ability_status_encoded"
    ]
    
    # Define the columns to be selected after transformation
    select_cols = ["bib_id", "FEATURES", "label"]

    # Use VectorAssembler  
    assembler = VectorAssembler(inputCols=index_cols, outputCol="FEATURES")

    # Transform both train and test datasets
    df_train = assembler.transform(df_train)
    df_test = assembler.transform(df_test)

    # Select the relevant columns for final output
    df_train_selected = df_train.select(select_cols)
    df_test_selected = df_test.select(select_cols)

    return df_train_selected, df_test_selected

def main():
    """
    Main function to load raw train and test data, apply preprocessing, and save the processed data.
    
    Args:
        train_path (str): Path to the raw train data.
        test_path (str): Path to the raw test data.
        processed_train_path (str): Path to save the processed train data.
        processed_test_path (str): Path to save the processed test data.
    """
    try:
        # Step 1: Load configuration
        config = load_config("config.json")
        read_path_template = get_config_value(config, "dataset", "path")
        write_path_template = get_config_value(config, "preprocessed", "path")


        # Step 2: Create Spark session
        spark = create_spark_session("Preprocess")
        spark.conf.set("spark.sql.parquet.datetimeRebaseModeInWrite", "CORRECTED")

        # Read raw train and test data
        df_train = read_parquet(spark=spark, parquet_path=read_path_template.format(type="train"))
        df_test = read_parquet(spark=spark, parquet_path=read_path_template.format(type="test"))

        # Apply preprocessing
        df_train, df_test = preprocess(df_train, df_test)
        logger.info("Process is done successfully")

        df_train, df_test = process_and_select_features(df_train=df_train, df_test=df_test)

        # Save the processed data
        write_parquet(df=df_train, output_parquet_path=write_path_template.format(type="train"))
        write_parquet(df=df_test, output_parquet_path=write_path_template.format(type="test"))

        logger.info("preprocess.py is finished successfully")

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
