import yaml 
import logging
from src.tools import tools

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

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
        import src.models.evaluate_model_text
    elif MODEL_TYPE == 'image':
        import src.models.evaluate_model_image
    elif MODEL_TYPE == 'merged':
        logger.info("Merged model evaluation is not implemented yet.")
        raise NotImplementedError("Merged model evaluation is not implemented yet.")