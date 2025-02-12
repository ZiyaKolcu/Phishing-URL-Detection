from PhishingDetection import logger
from PhishingDetection.pipeline.data_ingestion_pipeline import (
    DataIngestionTrainingPipeline,
)
from PhishingDetection.pipeline.data_validation_pipeline import (
    DataValidationTrainingPipeline,
)

STAGE_NAME = "Data Ingestion Stage"

try:
    logger.info(f">>>>> {STAGE_NAME} started <<<<<")
    data_ingestion = DataIngestionTrainingPipeline()
    data_ingestion.initiate_data_ingestion()
    logger.info(f">>>>> {STAGE_NAME} completed <<<<<")

except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Data Validation Stage"

try:
    logger.info(f">>>>> {STAGE_NAME} started <<<<<")
    data_validation = DataValidationTrainingPipeline()
    data_validation.initiate_data_validation()
    logger.info(f">>>>> {STAGE_NAME} completed <<<<<")

except Exception as e:
    logger.exception(e)
    raise e
