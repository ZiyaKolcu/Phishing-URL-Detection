from PhishingDetection import logger
from PhishingDetection.pipeline.data_ingestion_pipeline import (
    DataIngestionTrainingPipeline,
)
from PhishingDetection.pipeline.data_transformation_pipeline import (
    DataTransformationTrainingPipeline,
)
from PhishingDetection.pipeline.data_validation_pipeline import (
    DataValidationTrainingPipeline,
)
from PhishingDetection.pipeline.model_evaluation_pipeline import (
    ModelEvaluationTrainingPipeline,
)
from PhishingDetection.pipeline.model_trainer_pipeline import (
    ModelTrainerTrainingPipeline,
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

STAGE_NAME = "Data Transformation stage"

try:
    logger.info(f">>>>> {STAGE_NAME} started <<<<<")
    data_transformation = DataTransformationTrainingPipeline()
    data_transformation.initiate_data_transformation()
    logger.info(f">>>>> {STAGE_NAME} completed <<<<<")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Model Trainer stage"

try:
    logger.info(f">>>>> {STAGE_NAME} started <<<<<")
    model_trainer = ModelTrainerTrainingPipeline()
    model_trainer.initiate_model_trainer()
    logger.info(f">>>>> {STAGE_NAME} completed <<<<<")
except Exception as e:
    logger.exception(e)
    raise e


# STAGE_NAME = "Model Evaluation Stage"

# try:
#     logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
#     model_evaluation = ModelEvaluationTrainingPipeline()
#     model_evaluation.initiate_model_evaluation()
#     logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<")

# except Exception as e:
#     logger.exception(e)
#     raise e
