import sys
from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator

# Add the project path to sys.path
sys.path.append('/home/sajjad/Projects/ChurnApp')

# Import the functions to be executed
from src.ingest import main as ingest_main
from src.prepare_label import main as prepare_label_main

# Define the DAG
dag = DAG(
    'ingest_and_prepare_label_dag',
    start_date=datetime(2024, 9, 16),
    catchup=False,
    schedule_interval=None,
    tags=['churnapp', 'data-engineering']  # Tags for easier identification in Airflow UI
)

# Task 1: Run the ingest script
ingest_task = PythonOperator(
    task_id='ingest_data_task',
    python_callable=ingest_main,
    dag=dag
)

# Task 2: Run the prepare_label script
prepare_label_task = PythonOperator(
    task_id='prepare_label_task',
    python_callable=prepare_label_main,
    dag=dag
)

# Set task dependencies
ingest_task >> prepare_label_task