from src.embedder.ntv2 import NTv2Embedder
from src.embedder.resnet import ResNet1D

EMBEDDERS = {'NTv2': NTv2Embedder, 'ResNet1D': ResNet1D}


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