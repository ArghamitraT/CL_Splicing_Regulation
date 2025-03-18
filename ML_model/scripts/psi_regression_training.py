import sys
import os

# Add the parent directory (main) to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import hydra
from omegaconf import OmegaConf
import torch
from src.model.psi_regression import PSIRegressionModel
from src.trainer.utils import create_trainer
from src.datasets.auxiliary_jobs import PSIRegressionDataModule
from src.model.simclr import get_simclr_model
from src.utils.config import  print_config

@hydra.main(version_base=None, config_path="../configs", config_name="psi_regression.yaml")
def main(config: OmegaConf):

    # Register Hydra resolvers
    OmegaConf.register_new_resolver('eval', eval)
    OmegaConf.register_new_resolver('div_up', lambda x, y: (x + y - 1) // y)
    OmegaConf.register_new_resolver('min', lambda x, y: min([x, y]))
    OmegaConf.register_new_resolver('device_count', torch.cuda.device_count)

    # Print and process configuration
    print_config(config, resolve=True)

    # Initialize the IntronsDataModule
    data_module = PSIRegressionDataModule(config)

    data_module.prepare_data()
    data_module.setup()

    # Get the SimCLR model using its own config
    simclr_model = get_simclr_model(config)
    simclr_model.load_state_dict(torch.load("checkpoints/introns_cl/NTv2/199/best-checkpoint.ckpt")["state_dict"], strict=False)

    # Load weights from contrastive learning checkpoint
    simclr_ckpt = "checkpoints/introns_cl/NTv2/199/best-checkpoint.ckpt"
    state_dict = torch.load(simclr_ckpt)["state_dict"]

    # Decide based on Linear Probing vs. Fine-Tuning
    if config.model.freeze_encoder:  # Linear Probing Mode
        print("🚀 Running Linear Probing: Freezing encoder, keeping projection head.")
        simclr_model.load_state_dict(state_dict, strict=False)
        
        # Freeze encoder weights
        for param in simclr_model.encoder.parameters():
            param.requires_grad = False  

        model = PSIRegressionModel(simclr_model, config, freeze_encoder=True)

    else:  # Fine-Tuning Mode
        print("🔥 Running Fine-Tuning: Training the full encoder, using new head.")
        
        # Load only encoder weights, discard projection head
        encoder_state_dict = {k.replace("model.encoder.", ""): v for k, v in state_dict.items() if "model.encoder" in k}
        simclr_model.encoder.load_state_dict(encoder_state_dict, strict=False)

        model = PSIRegressionModel(simclr_model.encoder, config, freeze_encoder=False)

    # Create Trainer
    trainer = create_trainer(config)

    # Train the model
    trainer.fit(model, data_module.train_dataloader(), data_module.val_dataloader())


    #################


    # # Load pretrained encoder
    # encoder = get_simclr_model(config)
    # encoder.load_state_dict(torch.load("checkpoints/introns_cl/NTv2/199/best-checkpoint.ckpt")["state_dict"], strict=False)

    # # Choose training mode (Linear Probing = freeze encoder, Fine-Tuning = train encoder)
    # freeze_encoder = config.task.mode == "linear_probing"

    # # Create PSI Regression model
    # model = PSIRegressionModel(encoder, config, freeze_encoder=freeze_encoder)

    # # Create Trainer
    # trainer = create_trainer(config)

    # # Train the model
    # trainer.fit(model, data_module.train_dataloader(), data_module.val_dataloader())


if __name__ == "__main__":
    main()
