from src.backbone.ntv2 import NTv2Embedder

BACKBONES = {'NTv2': NTv2Embedder}


def get_backbone(config):
    """
    Function to get the backbone model for the task.

    Args:
        config (OmegaConf): The config object.

    Returns:
        nn.Module: The backbone model.
    """
    # Get the backbone model
    backbone = BACKBONES[config.backbone._name_](**config.backbone)
    return backbone