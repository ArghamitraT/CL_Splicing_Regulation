
from src.embedder.mtsplice.mtsplice import MTSpliceEncoder 


EMBEDDERS = {
    'MTSplice': MTSpliceEncoder 
}



def get_embedder(config):
    """
    Function to get the backbone model for the task.

    Args:
        config (OmegaConf): The config object.

    Returns:
        nn.Module: The backbone model.
    """
    # Get the backbone model
    embedder = EMBEDDERS[config.embedder._name_](**config.embedder)
    return embedder