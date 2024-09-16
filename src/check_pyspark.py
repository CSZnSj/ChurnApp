from pyspark.sql import SparkSession

def main():
    # Initialize a Spark session
    spark = SparkSession.builder \
        .appName("Airflow PySpark Integration") \
        .getOrCreate()

    # Sample data
    data = [("John", "Sales", 5000), ("Jane", "Finance", 6000), ("Mike", "IT", 7000)]
    columns = ["Name", "Department", "Salary"]

    # Create DataFrame
    df = spark.createDataFrame(data, columns)

    # Show the DataFrame
    df.show()

    # Perform some operations (example: filter where Salary > 5500)
    high_salary = df.filter(df.Salary > 5500)
    high_salary.show()

    # Stop the Spark session
    spark.stop()

if __name__ == "__main__":
    main()
