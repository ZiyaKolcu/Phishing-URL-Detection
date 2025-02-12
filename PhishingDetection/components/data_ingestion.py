import os
import sys
import pandas as pd
import psycopg2
from PhishingDetection import logger
from PhishingDetection.entity.config_entity import DataIngestionConfig
from dotenv import load_dotenv

load_dotenv()


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        self.data_ingestion_config = data_ingestion_config

    def export_database_table_as_dataframe(self):
        """
        Read data from PosgreSQL
        """
        try:
            database_name = os.getenv("DATABASE_NAME")
            database_username = os.getenv("DATABASE_USERNAME")
            database_password = os.getenv("DATABASE_PASSWORD")
            database_table_name = os.getenv("DATABASE_TABLE_NAME")

            conn = psycopg2.connect(
                database=database_name,
                user=database_username,
                password=database_password,
            )

            query = f"SELECT * FROM {database_table_name};"
            df = pd.read_sql(query, conn)

            if "id" in df.columns.to_list():
                df = df.drop(columns=["id"], axis=1)
            conn.close()

            logger.info("Data exported successfully!")

            return df

        except Exception as e:
            logger.exception(e)
            raise e

    def export_data_into_file(self, dataframe: pd.DataFrame):
        try:
            data_dir = self.data_ingestion_config.local_data_dir

            if not os.path.exists(data_dir):
                os.makedirs(data_dir, exist_ok=True)

            file_path = os.path.join(data_dir, "data.csv")

            dataframe.to_csv(file_path, index=False, header=True)

            logger.info(f"Data saved successfully at {file_path}")

            return dataframe

        except Exception as e:
            logger.exception(e)
            raise e
