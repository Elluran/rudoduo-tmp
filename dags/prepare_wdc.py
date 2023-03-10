from airflow import Dataset
from airflow.decorators import dag, task
import pendulum
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_preparation import unpack_wdc


raw_wdc = Dataset("raw_wdc")
extracted_ru_wdc_tables = Dataset("extracted_ru_wdc_tables")


@dag(
    catchup=False,
    start_date=pendulum.datetime(2021, 1, 1, tz="UTC"),
    tags=["filtering"],
    schedule=[raw_wdc]
)
def extract_ru_wdc_tables():
    """
    """
    @task(outlets=[extracted_ru_wdc_tables])
    def extract_tables():
        """
        """
        return unpack_wdc.main()

    extract_tables()

extract_ru_wdc_tables()
