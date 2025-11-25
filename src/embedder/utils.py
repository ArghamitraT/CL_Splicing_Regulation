
from src.embedder.mtsplice.mtsplice import MTSpliceEncoder 
from src.embedder.dilated_conv_1d import DilatedConvEmbedder


EMBEDDERS = {
    'MTSplice': MTSpliceEncoder,
    'DilatedConv1D': DilatedConvEmbedder
}



def get_embedder(config):
    """
    Function to get the backbone model for the task.

    Args:
        config (OmegaConf): The config object.

    Returns:
        nn.Module: The backbone model.
    """
    embedder = EMBEDDERS[config.embedder._name_](**config.embedder)
    return embedder
