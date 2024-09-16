# dags/pyspark_dag.py

from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

# Define the default arguments
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 9, 16),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define the DAG
dag = DAG(
    'check_pyspark_dag',
    default_args=default_args,
    description='A simple DAG to run a PySpark job',
    schedule_interval=timedelta(days=1),
)

# Task to run the PySpark job
run_pyspark_job = BashOperator(
    task_id='run_pyspark_job',
    bash_command='spark-submit home/sajjad/Projects/ChurnApp/src/check_pyspark.',
    dag=dag,
)

# Define the task pipeline
run_pyspark_job
