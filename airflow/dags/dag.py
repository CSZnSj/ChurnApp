# dag.py

import sys
sys.path.append('/home/sajjad/Projects/ChurnApp')

from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator

from src.ingest import main as ingest_main
from src.prepare_label import main as prepare_label_main
from src.prepare_dataset import main as prepare_dataset_main

# Define the DAG
dag = DAG(
    'flow_dag',
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

# Task 3: Run the prepare_dataset script
prepare_dataset_task = PythonOperator(
    task_id='prepare_dataset_task',
    python_callable=prepare_dataset_main,
    dag=dag
)

# Set task dependencies
ingest_task >> prepare_label_task >> prepare_dataset_task