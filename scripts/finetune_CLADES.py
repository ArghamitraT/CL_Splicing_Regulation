import sys
import os
import subprocess
import torch
from pathlib import Path
from omegaconf import OmegaConf

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def find_contrastive_root(start: Path = Path(__file__)) -> Path:
    for parent in start.resolve().parents:
        if parent.name == "CLADES":
            return parent
    raise RuntimeError("Could not find 'CLADES' directory.")

root_path = str(find_contrastive_root())
os.environ["CONTRASTIVE_ROOT"] = root_path


import hydra
from omegaconf import OmegaConf
import torch
from src.trainer.utils import create_trainer
from src.datasets.auxiliary_jobs import PSIRegressionDataModule
from src.utils.config import  print_config
from src.utils.encoder_init import initialize_encoders_and_model


import importlib


def load_model_class(config):
    module_name = f"src.model.{config.aux_models.model_script}"
    module = importlib.import_module(module_name)
    return getattr(module, "PSIRegressionModel")

def get_optimal_num_workers():
    num_cpus = os.cpu_count()
    num_gpus = torch.cuda.device_count()
    return min(num_cpus // max(1, num_gpus), 16)


def count_parameters(model):
    print(model)
    print("\n" + "="*80)
    print(f"{'Layer':40s} {'Param #':>15s}")
    print("="*80)
    
    total_params = 0
    for name, param in model.named_parameters():
        num_params = param.numel()
        print(f"{name:40s} {num_params:15,}")
        total_params += num_params
    
    print("="*80)
    print(f"{'Total parameters:':40s} {total_params:15,}")
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{'Trainable parameters:':40s} {trainable_params:15,}")
    print(f"{'Non-trainable parameters:':40s} {total_params - trainable_params:15,}")


@hydra.main(version_base=None, config_path="../configs", config_name="finetune_CLADES.yaml")
def main(config: OmegaConf):

    def get_free_gpu():
        if not torch.cuda.is_available():
            print("CUDA is not available. Running on CPU.")
            return None
        try:
            # Try using nvidia-smi via full path or direct shell
            result = subprocess.check_output(
                "nvidia-smi --query-gpu=memory.used --format=csv,nounits,noheader", 
                shell=True
            )
            memory_used = [int(x) for x in result.decode("utf-8").strip().split("\n")]
            return memory_used.index(min(memory_used))
        except Exception as e:
            print(f"Warning: Could not query nvidia-smi ({e}). Using GPU 0 by default.")
            return 0

    # Choose and set GPU
    free_gpu = get_free_gpu()
    if free_gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(free_gpu)
        print(f"Using GPU {free_gpu}: {torch.cuda.get_device_name(0)}")
    else:
        print("Running on CPU")


    # Register Hydra resolvers
    OmegaConf.register_new_resolver('eval', eval)
    OmegaConf.register_new_resolver('div_up', lambda x, y: (x + y - 1) // y)
    OmegaConf.register_new_resolver('min', lambda x, y: min([x, y]))
    OmegaConf.register_new_resolver('device_count', torch.cuda.device_count)
    OmegaConf.register_new_resolver('optimal_workers', lambda: get_optimal_num_workers())

    
    # Print and process configuration
    print_config(config, resolve=True)

    # Initialize the IntronsDataModule 
    data_module = PSIRegressionDataModule(config)
    data_module.prepare_data()
    data_module.setup()

    # EVAL-ONLY: load checkpoint and run test
    mode = str(getattr(config.aux_models, "train_mode", "train")).lower()
    ckpt_path = getattr(config.aux_models, "eval_weights", None)
    
    if mode == "eval":
        config.aux_models.warm_start = False
        model = initialize_encoders_and_model(config, root_path)
        print(f"[Eval] Loading checkpoint: {ckpt_path}")
        ckpt_path = f"{root_path}/output/{ckpt_path}/checkpoints/{config.task._name_}/{config.embedder._name_}/{config.dataset.seq_len}/best-checkpoint.ckpt"
        trainer = create_trainer(config)   
        trainer.test(model=model, ckpt_path=ckpt_path, datamodule=data_module)

    else:
        model = initialize_encoders_and_model(config, root_path)
        model = model.float()
        count_parameters(model)

        trainer = create_trainer(config)
        trainer.fit(model, data_module.train_dataloader(), data_module.val_dataloader())
        trainer.test(model, datamodule=data_module)

    print('######END#######')
    

if __name__ == "__main__":
    main()



    