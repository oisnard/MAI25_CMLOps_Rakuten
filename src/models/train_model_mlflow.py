import yaml 
import logging
from src.tools import tools
import argparse


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="Train a model using MLflow")
    parser.add_argument('--pipeline', 
                        type=str, 
                        default='full', 
                        help='Full pipeline on all available data or datastream for emulating incoming data streams')
    args = parser.parse_args()
    pipeline_mode = args.pipeline.lower()

    if pipeline_mode not in ['full', 'datastream']:
        logger.error(f"Invalid pipeline mode: {pipeline_mode}. It should be one of ['full', 'datastream'].")
        raise ValueError(f"Invalid pipeline mode: {pipeline_mode}. It should be one of ['full', 'datastream'].")
    logger.info(f"Pipeline mode selected: {pipeline_mode}")

    # Load parameters from YAML file
    try:
        # Load dataset parameters from YAML file
        params = tools.load_dataset_params_from_yaml()    
    except Exception as e:
        logger.error(f"An unexpected error occurred when loading params.yaml file: {e}")
        raise

    # Load the model type from params
    MODEL_TYPE = params['model_selection']['model_type']

    if MODEL_TYPE not in ['text', 'image', 'merged']:
        logger.error(f"Invalid model type: {MODEL_TYPE}. It should be one of ['text', 'image', 'merged'].")
        raise ValueError(f"Invalid model type: {MODEL_TYPE}. It should be one of ['text', 'image', 'merged'].")
    logger.info(f"Model type selected: {MODEL_TYPE}")
    if MODEL_TYPE == 'text':
        from src.models.train_model_text_mlflow import main as train_text_model
        train_text_model(pipeline_mode=pipeline_mode)
    elif MODEL_TYPE == 'image':
        from src.models.train_model_image_mlflow import main as train_image_model
        train_image_model(pipeline_mode=pipeline_mode)
    elif MODEL_TYPE == 'merged':
        from src.models.train_model_both_mlflow import main as train_both_model
        train_both_model(pipeline_mode=pipeline_mode)
    else:
        raise NotImplementedError(f"MODEL TYPE {MODEL_TYPE} not implemented. Please check the model_selection in params.yaml.")