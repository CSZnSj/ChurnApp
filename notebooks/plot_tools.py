from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import lit, to_timestamp, col, sum, count, hour, dayofweek, dayofmonth, when, datediff
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Optional


def save_fig(
        name: str, 
        output_dir: str, 
        file_format: str = "png") -> None:
    """
    Saves the current figure to a specified directory with a given name and file format.

    Args:
        output_dir (str): The directory where the figure will be saved.
        name (str): The name of the file (without extension) for the saved figure.
        file_format (str, optional): The file format to use for saving the figure. Defaults to "png".

    Raises:
        ValueError: If an unsupported file format is specified.

    Returns:
        None
    """
    # Validate file format
    supported_formats = ["png", "jpg", "jpeg", "pdf", "svg"]
    if file_format not in supported_formats:
        raise ValueError(f"Unsupported file format '{file_format}'. Supported formats are: {supported_formats}.")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Construct file path
    file_path = os.path.join(output_dir, f"{name}.{file_format}")

    # Save the 
    plt.title(name)
    plt.subplots_adjust(bottom=0.3)
    plt.tight_layout()
    plt.savefig(file_path, format=file_format)
    plt.close()

    # Log the success message
    print(f"Figure saved to: {file_path}")

def plot_distribution(
        df: DataFrame, 
        plot_type: str,
        month: str,
        x: str, 
        y: str = None, 
        hue: str = None, 
        figsize: tuple = (12, 8), 
        output_dir: str = "plots") -> None:
    """
    Plots the distribution of a specified column in a DataFrame using various plot types.

    Args:
        col_df (pd.DataFrame): The DataFrame containing the column to plot.
        col_name (str): The name of the column to plot.
        plot_type (str): The type of plot to create ('box', 'hist', 'violin', 'count').
        figsize (tuple, optional): Figure size for the plot. Defaults to (12, 8).
        hue (str, optional): Variable in `df` to map plot aspects to different colors. Defaults to None.

    Returns:
        None
    """
    # Check if the plot type is supported
    supported_plot_types = {
        "count": sns.countplot,
        "line": sns.lineplot,
        "box": sns.boxplot,
        "hist": sns.histplot,
        "kde": sns.kdeplot,
        "violin": sns.violinplot,
    }
    
    if plot_type not in supported_plot_types:
        raise ValueError(f"Unsupported plot type '{plot_type}'. Supported types are: {list(supported_plot_types.keys())}.")

    # Convert PySpark DataFrame to Pandas DataFrame if necessary
    if isinstance(df, DataFrame):
        df = df.toPandas()
    elif isinstance(df, pd.DataFrame):
        df = df
    else:
        raise TypeError("The provided DataFrame is neither a PySpark DataFrame nor a Pandas DataFrame.")

    # Set up the plot
    plt.figure(figsize=figsize)
    plot_func = supported_plot_types[plot_type]
    if y:
        plot_func(data=df, x=x, y=y, hue=hue)
    else:
        plot_func(data=df, x=x, hue=hue)

    name = f"{plot_type.capitalize()}Plot_of_{x}__month_{month}"
    if hue:
        name += "__is_hued"
    plt.xticks(rotation=90)
    save_fig(name=name, output_dir=output_dir)


def plot_aggregated_by_time(
        df: DataFrame, 
        timestamp_column: str, 
        agg_col : str, 
        month: str,
        figsize: Tuple = (12, 8), 
        output_dir : str = "plots") -> None:
    """
    Aggregates loan amounts based on different time components and plots the results.

    Args:
        dataframe (DataFrame): A PySpark DataFrame containing the data to be aggregated.
        timestamp_column (str): The name of the timestamp column to be used for aggregation.
        month (str): A string representing the month, used for naming output files.

    Returns:
        None
    """
    
    # Define keys for time aggregation and corresponding functions
    time_keys = {
        'Hour': hour,
        'DayOfWeek': dayofweek,
        'DayOfMonth': dayofmonth
    }

    # Add time-based columns to the DataFrame
    for key, time_function in time_keys.items():
        df = df.withColumn(key, time_function(col(timestamp_column)))

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Iterate over each key for aggregation and plotting
    for key in time_keys.keys():

        # Aggregate the loan amounts by the current time key
        aggregated_df = (
            df.groupBy(key)
            .agg(sum(agg_col).alias(f"sum_{agg_col}_{month}"))
            .toPandas()
        )

        # Prepare the DataFrame for plotting
        aggregated_df.reset_index(drop=True, inplace=True)
        name = f"SumAggregated_{agg_col}_by_{key}"
        aggregated_df[name] = aggregated_df.index

        plot_distribution(df=aggregated_df, plot_type="line", month=month, x=name, y=f"sum_{agg_col}_{month}", figsize=figsize)

def categorize(
    df: DataFrame,
    col_name: str,
    bins: Optional[List[float]],
    labels: Optional[List[str]],
    new_column_name: str = "category"
) -> DataFrame:
    """
    Categorizes a numeric column in a PySpark DataFrame into bins with descriptive labels.

    Args:
        df (DataFrame): The input PySpark DataFrame.
        column_name (str): The name of the column to categorize.
        bins (List[float]): List of bin edges. Should be sorted in ascending order and include -inf and inf.
        labels (List[str], optional): List of labels corresponding to the bins. Should have one less element than bins.
        new_column_name (str, optional): The name of the new column that will contain the category labels.
            Defaults to "category".

    Returns:
        DataFrame: A new DataFrame with the added category column.

    Raises:
        ValueError: If the number of bins does not match the number of labels + 1.
    """

    # Validate bin and label lengths
    if len(bins) != len(labels) + 1:
        raise ValueError("The number of bins must be one more than the number of labels.")
    if not bins or not labels:
        raise ValueError("Bins and labels cannot be empty.")

    # Start with a column containing the original values
    categorized_df = df.withColumn(new_column_name, col(col_name))

    # Apply categorization logic
    for i in range(1, len(bins)):
        categorized_df = categorized_df.withColumn(
            new_column_name,
            when((col(col_name) > bins[i - 1]) & (col(col_name) <= bins[i]), labels[i - 1])
            .otherwise(col(new_column_name))
        )

    return categorized_df

def categorize_time_difference(
    df: DataFrame,
    activation_col: str = "activation_date",
    deactivation_col: str = "deactivation_date",
    new_col_name: str = "time_difference_days"
) -> DataFrame:
    """
    Calculates the time difference between activation and deactivation dates in days,
    and categorizes this difference into bins.

    Args:
        df (DataFrame): Input PySpark DataFrame.
        activation_col (str): Column name for activation date.
        deactivation_col (str): Column name for deactivation date.
        new_col_name (str): Name of the new column for time difference in days.

    Returns:
        DataFrame: DataFrame with time difference in days and categorized time difference.
    """
    # Calculate the time difference in days
    df = df.withColumn(new_col_name, datediff(col(deactivation_col), col(activation_col)))

    # Define bins and labels for categorization
    bins = [-float('inf'), 0, 1, 7, 15, 30, 60, 120, 180, 365, float('inf')]
    labels = ["Same Day", "Daily", "Weekly", "Half-Monthly", "Monthly", "2 Months", "4 Months", "6 Months", "1 Year", "More than 1 Year"]

    # Categorize the time differences using the categorize function
    df = categorize(df, col_name=new_col_name, bins=bins, labels=labels, new_column_name="duration_category")

    return df

def plot_tops(
        df: DataFrame,
        col_name: str,
        index: int,
        month: int,
        figsize: Tuple = (12, 8), 
        output_dir: str = "plots") -> None:
    """
    Groups the DataFrame by 'offering_name', counts occurrences, sorts them in descending order,
    plots the top 20 most popular offerings, and saves the plot to the specified output path.

    Args:
        package (DataFrame): Input DataFrame with an 'offering_name' column.
        output_path (str): The path to save the output plot.
    
    Returns:
        None
    """
    # Group by 'offering_name', aggregate with count, and order by count descending
    result_df = (
        df
        .groupBy(col_name)
        .agg(count("*").alias("count"))
        .orderBy("count", ascending=False)
    )

    # Convert the result to pandas DataFrame for plotting
    top = result_df.limit(index).toPandas()

    # Plotting the top packages
    plt.figure(figsize=figsize)
    plt.bar(top[col_name], top["count"], color='skyblue')
    plt.xticks(rotation=45, ha='right')
    plt.xlabel(col_name)
    plt.ylabel("Count")
    name = f"top_{index}_popular_{col_name}s__month_{month}"
    save_fig(name=name, output_dir=output_dir)


def plot_analysis_of_loan_recovery(
        assign: DataFrame, 
        recovery: DataFrame,
        figsize: Tuple = (12, 8), 
        output_dir = "plots") -> None:
    """
    Analyzes the loan recovery data by calculating the total loan amounts and recoveries, 
    and computes the percentage of loans that are fully paid.

    Parameters:
    - assign (DataFrame): The DataFrame containing assigned loans.
    - recovery (DataFrame): The DataFrame containing loan recovery data.

    Returns:
    - tuple: A tuple containing (recovered_sum, assigned_sum, recovery_ratio, percentage_of_people)
    """
    
    # Find common loan IDs between assign and recovery DataFrames
    common_loan_ids = assign.select("loan_id").intersect(recovery.select("loan_id"))

    # Join recovery DataFrame with common loan IDs
    df = recovery.join(common_loan_ids, on="loan_id", how="inner")

    # Group by loan_id and aggregate sum of loan_amount and hsdp_recovery
    df = df.groupBy("loan_id").agg(
        sum(col("loan_amount")).alias("sum_of_loan_amount"),
        sum(col("hsdp_recovery")).alias("sum_of_recovery")
    )

    # Calculate the total assigned sum and recovered sum
    recovered_sum = df.select(sum(col("sum_of_recovery"))).collect()[0][0]
    assigned_sum = df.select(sum(col("sum_of_loan_amount"))).collect()[0][0]
    percentage_of_recovered = recovered_sum / assigned_sum * 100

    # Save a plot as an image
    plt.figure(figsize=figsize)
    bars = plt.bar(['RecoveredSum', 'AssignedSum'], [recovered_sum, assigned_sum], color=['green', 'red'])
    # Annotate the bars with the sum values
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval, np.round(yval, 2), ha='center', va='bottom')
    plt.ylabel(f'Amount')
    name = f'Comparison of Recovered and Assigned Loan Amounts --- {percentage_of_recovered:.2f}% of loan_amounts are recovered'
    save_fig(name=name, output_dir=output_dir)

