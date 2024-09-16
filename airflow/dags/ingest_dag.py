import sys
sys.path.append('/home/sajjad/Projects/ChurnApp')

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from src.ingest import main

# Define the DAG
with DAG(
    'ingest_dag',
    start_date=datetime(2024, 9, 16),
    catchup=False,
    schedule_interval=None,
) as dag:
    
    ingest_task = PythonOperator(
        task_id='ingest_data_task',
        python_callable=main,
    )