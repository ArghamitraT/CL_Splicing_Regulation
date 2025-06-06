import sys
import os
import subprocess
import torch
from pathlib import Path
from omegaconf import OmegaConf

# Add the parent directory (main) to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def find_contrastive_root(start: Path = Path(__file__)) -> Path:
    for parent in start.resolve().parents:
        if parent.name == "Contrastive_Learning":
            return parent
    raise RuntimeError("Could not find 'Contrastive_Learning' directory.")

# Set env var *before* hydra loads config
# os.environ["CONTRASTIVE_ROOT"] = str(find_contrastive_root())
root_path = str(find_contrastive_root())
os.environ["CONTRASTIVE_ROOT"] = root_path


import hydra
from omegaconf import OmegaConf
import torch
# from src.model.psi_regression import PSIRegressionModel
from src.trainer.utils import create_trainer
from src.datasets.auxiliary_jobs import PSIRegressionDataModule
# from src.model.simclr import get_simclr_model
from src.utils.config import  print_config
from src.utils.encoder_init import initialize_encoders_and_model


import importlib

def load_model_class(config):
    module_name = f"src.model.{config.aux_models.model_script}"
    module = importlib.import_module(module_name)
    return getattr(module, "PSIRegressionModel")

# os.environ['WANDB_INIT_TIMEOUT'] = '600'
def get_optimal_num_workers():
    num_cpus = os.cpu_count()
    num_gpus = torch.cuda.device_count()
    return min(num_cpus // max(1, num_gpus), 16)




@hydra.main(version_base=None, config_path="../configs", config_name="psi_regression.yaml")
def main(config: OmegaConf):

    def get_free_gpu():
        result = subprocess.check_output(
            "nvidia-smi --query-gpu=memory.used --format=csv,nounits,noheader", shell=True
        )
        memory_used = [int(x) for x in result.decode("utf-8").strip().split("\n")]
        return memory_used.index(min(memory_used))

    # Choose and set GPU
    free_gpu = get_free_gpu()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(free_gpu)
    print(f"Using GPU {free_gpu}: {torch.cuda.get_device_name(0)}")


    # Register Hydra resolvers
    OmegaConf.register_new_resolver('eval', eval)
    OmegaConf.register_new_resolver('div_up', lambda x, y: (x + y - 1) // y)
    OmegaConf.register_new_resolver('min', lambda x, y: min([x, y]))
    OmegaConf.register_new_resolver('device_count', torch.cuda.device_count)
    OmegaConf.register_new_resolver('optimal_workers', lambda: get_optimal_num_workers())

    # result_dir = config.task.pretraining_weights

    
    # Print and process configuration
    print_config(config, resolve=True)

    # Initialize the IntronsDataModule  #####(AT)######
    data_module = PSIRegressionDataModule(config)
    data_module.prepare_data()
    data_module.setup()
    print()





    # # Get the SimCLR model using its own configß
    # simclr_model = get_simclr_model(config)

    # if config.aux_models.warm_start:
    #     # simclr_model.load_state_dict(torch.load("checkpoints/introns_cl/NTv2/199/best-checkpoint.ckpt")["state_dict"], strict=False)
    #     # simclr_ckpt = "/mnt/home/at3836/Contrastive_Learning/files/results/exprmnt_2025_05_04__11_29_05/weights/checkpoints/introns_cl/ResNet1D/199/best-checkpoint.ckpt"
        
    #     simclr_ckpt = f"{root_path}/files/results/{result_dir}/weights/checkpoints/introns_cl/{config.embedder._name_}/199/best-checkpoint.ckpt"

    #     ckpt = torch.load(simclr_ckpt)
    #     state_dict = ckpt["state_dict"]

    #     # REMOVE "model." prefix from all keys
    #     cleaned_state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}

    #     missing, unexpected = simclr_model.load_state_dict(cleaned_state_dict, strict=False)
    
    # # Instantiate PSIRegressionModel with encoder and psi_model config
    # # model = PSIRegressionModel(simclr_model.encoder, config)
    # PSIRegressionModel = load_model_class(config)
    # model = PSIRegressionModel(encoder_5p, encoder_3p, encoder_exon, config)
    model = initialize_encoders_and_model(config, root_path)

    


    # Traingi
    trainer = create_trainer(config)
    trainer.fit(model, data_module.train_dataloader(), data_module.val_dataloader())
    trainer.test(model, datamodule=data_module)
    

if __name__ == "__main__":
    main()



    