""" Utils for the training loop.

Copied from "https://github.com/kuleshov-group/caduceus"
"""

import json
import logging
import warnings

import rich.syntax
import rich.tree
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities import rank_zero_only

from src.utils.config import omegaconf_filter_keys
import fsspec


# Copied from https://docs.python.org/3/howto/logging-cookbook.html#using-a-context-manager-for-selective-logging
class LoggingContext:
    def __init__(self, logger, level=None, handler=None, close=True):
        self.logger = logger
        self.level = level
        self.handler = handler
        self.close = close

    def __enter__(self):
        if self.level is not None:
            self.old_level = self.logger.level
            self.logger.setLevel(self.level)
        if self.handler:
            self.logger.addHandler(self.handler)

    def __exit__(self, et, ev, tb):
        if self.level is not None:
            self.logger.setLevel(self.old_level)
        if self.handler:
            self.logger.removeHandler(self.handler)
        if self.handler and self.close:
            self.handler.close()
        # implicit return of None => don't swallow exceptions


def get_logger(name=__name__, level=logging.INFO) -> logging.Logger:
    """Initializes multi-GPU-friendly python logger.x"""

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    for level in ("debug", "info", "warning", "error", "exception", "fatal", "critical"):
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


def process_config(config: DictConfig) -> DictConfig:  # TODO because of filter_keys, this is no longer in place
    """A couple of optional utilities, controlled by main config file:
    - disabling warnings
    - easier access to debug mode
    - forcing debug friendly configuration
    Modifies DictConfig in place.
    Args:
        config (DictConfig): Configuration composed by Hydra.
    """
    log = get_logger()

    # Filter out keys that were used just for interpolation
    config = omegaconf_filter_keys(config, lambda k: not k.startswith('__'))

    # enable adding new keys to config
    OmegaConf.set_struct(config, False)

    # disable python warnings if <config.ignore_warnings=True>
    if config.get("ignore_warnings"):
        log.info("Disabling python warnings! <config.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    if config.get("debug"):
        log.info("Running in debug mode! <config.debug=True>")
        config.trainer.fast_dev_run = True

        # force debugger friendly configuration
        log.info("Forcing debugger friendly configuration! <config.trainer.fast_dev_run=True>")
        # Debuggers don't like GPUs or multiprocessing
        if config.trainer.get("gpus"):
            config.trainer.gpus = 0
        if config.loader.get("pin_memory"):
            config.loader.pin_memory = False
        if config.loader.get("num_workers"):
            config.loader.num_workers = 0

    # disable adding new keys to config
    # OmegaConf.set_struct(config, True) # [21-09-17 AG] I need this for .pop(_name_) pattern among other things

    return config


@rank_zero_only
def print_config(
        config: DictConfig,
        resolve: bool = True,
        save_cfg=True,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.
    Args:
        config (DictConfig): Configuration composed by Hydra.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
        save_cfg (bool, optional): Whether to save the config to a file.
    """

    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    fields = config.keys()
    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)

        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, DictConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    rich.print(tree)

    if save_cfg:
        with open("config_tree.txt", "w") as fp:
            rich.print(tree, file=fp)
        with open("model_config.json", "w") as fp:  # Save config / model config for use in fine-tuning or testing
            model_config = {
                k: v
                for k, v in OmegaConf.to_container(config.model, resolve=True).items()
                if not k.startswith("_") or k == "config_path"
            }
            json.dump(model_config, fp, indent=4)
        with open("config.json", "w") as fp:
            json.dump(OmegaConf.to_container(config, resolve=True), fp, indent=4)


def log_optimizer(logger, optimizer, keys):
    """ Log values of particular keys from the optimizers param groups """
    keys = sorted(keys)
    for i, g in enumerate(optimizer.param_groups):
        group_hps = {k: g.get(k, None) for k in keys}
        logger.info(' | '.join([
                                   f"Optimizer group {i}",
                                   f"{len(g['params'])} tensors",
                               ] + [f"{k} {v}" for k, v in group_hps.items()]))


class OptimModule(nn.Module):
    """ Interface for Module that allows registering buffers/parameters with configurable optimizer hyperparameters """

    def register(self, name, tensor, lr=None, wd=0.0):
        """Register a tensor with a configurable learning rate and 0 weight decay"""

        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))

            optim = {}
            if lr is not None:
                optim["lr"] = lr
            if wd is not None:
                optim["weight_decay"] = wd
            setattr(getattr(self, name), "_optim", optim)
            
# Check if file exists using fsspec
def fsspec_exists(filename):
    """Check if file exists in manner compatible with fsspec."""
    fs, _ = fsspec.core.url_to_fs(filename)
    return fs.exists(filename)

# List directory contents using fsspec
def fsspec_listdir(dirname):
    """Listdir in manner compatible with fsspec."""
    fs, _ = fsspec.core.url_to_fs(dirname)
    return fs.ls(dirname)


import inspect
import os
_warned_debug = False  # Define the global flag once
def debug_warning(message="Using fixed species views! REMEMBER TO REVERT!",
                  file_override=None, line_override=None):
    global _warned_debug
    if not _warned_debug:
        if file_override and line_override:
            filename, lineno = file_override, line_override
        else:
            frame = inspect.currentframe().f_back
            filename = os.path.basename(frame.f_code.co_filename)
            lineno = frame.f_lineno
        print(f"\033[1;31m⚠️ DEBUG WARNING in {filename}:{lineno} — {message}\033[0m")
        _warned_debug = True

def reset_debug_warning():
    global _warned_debug
    _warned_debug = False
