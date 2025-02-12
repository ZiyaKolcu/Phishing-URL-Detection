from PhishingDetection import logger
from PhishingDetection.pipeline.data_ingestion_pipeline import (
    DataIngestionTrainingPipeline,
)

STAGE_NAME = "Data Ingestion Stage"

if __name__ == "__main__":
    try:
        logger.info(f">>>>> {STAGE_NAME} started <<<<<")
        data_ingestion = DataIngestionTrainingPipeline()
        data_ingestion.initiate_data_ingestion()
        logger.info(f">>>>> {STAGE_NAME} completed <<<<<")
    except Exception as e:
        logger.exception(e)
        raise e
