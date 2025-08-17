import torch
import torch.nn as nn
import lightning.pytorch as pl
from hydra.utils import instantiate
from torchmetrics import R2Score
import time
from scipy.stats import spearmanr
from scipy.special import logit
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class MTSpliceBCE(pl.LightningModule):
    def __init__(self, encoder, config, embed_dim=32, out_dim=56, dropout=0.5):
        super().__init__()
        # self.save_hyperparameters(ignore=['encoder'])

        self.encoder = encoder
        self.config = config

        if self.config.aux_models.freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False


        if hasattr(encoder, "get_last_embedding_dimension") and callable(encoder.get_last_embedding_dimension):
            print("📏 Using encoder.get_last_embedding_dimension()")
            encoder_output_dim = encoder.get_last_embedding_dimension()

        else:
            print("⚠️ Warning: `encoder.output_dim` not defined, falling back to dummy input.")
            if hasattr(config, "dataset") and hasattr(config.dataset, "seq_len"):
                seq_len = config.dataset.seq_len
            else:
                raise ValueError("`seq_len` not found in config.dataset — can't create dummy input.")

            dummy_input = torch.full((1, 4, seq_len), 1.0)  # one-hot-style dummy input
            dummy_input = dummy_input.to(next(encoder.parameters()).device)

            with torch.no_grad():
                dummy_output = encoder(dummy_input)
                encoder_output_dim = dummy_output.shape[-1]

            print(f"📏 Inferred encoder output_dim = {encoder_output_dim}")
        
        self.fc1 = nn.Linear(encoder_output_dim, embed_dim)
        self.bn2 = nn.BatchNorm1d(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(embed_dim, out_dim)


        # Instantiate loss and metrics via Hydra
        self.loss_fn = instantiate(config.loss)

        self.metric_fns = []
        for metric in config.task.metrics:
            if metric == "r2_score":
                self.metric_fns.append(R2Score())

    
    def forward(self, x):
        
        seql, seqr = x
        features = self.encoder(seql, seqr)
        x = self.fc1(features)
        x = self.bn2(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x    
        

    def training_step(self, batch, batch_idx):
        x, y, exon_ids = batch
        y_pred = self(x).squeeze()
        
        if self.config.aux_models.mtsplice_BCE:
            loss = self.loss_fn(y_pred, exon_ids, split='train')
        else:
            loss = self.loss_fn(y_pred, y)

        self.log("train_loss", loss, on_epoch=True, on_step=True, prog_bar=True, sync_dist=True)
        for metric_fn in self.metric_fns:
            if self.config.aux_models.mtsplice_BCE:
                break
            self.log(f"train_{metric_fn.__class__.__name__}", metric_fn(y_pred, y), on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        
        x, y, exon_ids = batch
        y_pred = self(x).squeeze()
        
        if self.config.aux_models.mtsplice_BCE:
            loss = self.loss_fn(y_pred, exon_ids, split='val')
        else:
            loss = self.loss_fn(y_pred, y)

        self.log("val_loss", loss, on_epoch=True, on_step=True, prog_bar=True, sync_dist=True)
        for metric_fn in self.metric_fns:
            if self.config.aux_models.mtsplice_BCE:
                break
            self.log(f"val_{metric_fn.__class__.__name__}", metric_fn(y_pred, y), on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y, exon_ids = batch
        y_pred = self(x).squeeze()
        
        if self.config.aux_models.mtsplice_BCE:
            split = self.config.dataset.test_files.intronexon.split('/')[-1].split('_')[1]
            loss = self.loss_fn(y_pred, exon_ids, split)
        else:
            loss = self.loss_fn(y_pred, y)

        self.log("test_loss", loss, on_epoch=True, on_step=True, prog_bar=True, sync_dist=True)
        for metric_fn in self.metric_fns:
            if self.config.aux_models.mtsplice_BCE:
                break
            self.log(f"test_{metric_fn.__class__.__name__}", metric_fn(y_pred, y), on_epoch=True, prog_bar=True, sync_dist=True)

        # Store for Spearman
        self.test_preds.append(y_pred.detach().cpu())
        self.test_targets.append(y.detach().cpu())

        # === NEW: Store exon IDs
        self.test_exon_ids += list(exon_ids)

        return loss

    def on_test_epoch_start(self):
        self.test_preds = []
        self.test_targets = []
        self.test_exon_ids = []

    def on_test_epoch_end(self):

        if self.config.aux_models.mtsplice_BCE:

            from scipy.special import logit, expit

            # === Setup
            tissue = "Retina - Eye"
            tissue_index = 0  # position of Retina - Eye in the 56 tissues

            # === Load prediction results
            # Assume you have:
            #   result: tensor of shape [N, 56]
            #   exon_ids: list of length N
            y_pred = torch.cat(self.test_preds, dim=0).numpy()         # shape: [N, 56]
            exon_ids = self.test_exon_ids   
            retina_pred = y_pred[:, tissue_index]  # shape: [N]

            # Build prediction DataFrame
            predictions = pd.DataFrame({
                'exon_id': exon_ids,
                tissue: retina_pred
            })
            split = self.config.dataset.test_files.intronexon.split('/')[-1].split('_')[1]
            # === Load ground truth
            ground_truth = pd.read_csv(f"/mnt/home/at3836/Contrastive_Learning/data/final_data/ASCOT_finetuning/{split}_cassette_exons_with_logit_mean_psi.csv")

            # === Merge prediction and ground truth
            df = pd.merge(
                ground_truth[['exon_id', 'logit_mean_psi', tissue]],
                predictions[['exon_id', tissue]],
                on='exon_id',
                suffixes=('_true', '_logit_delta_pred')
            )

            # === Compute final predicted PSI
            df['final_predicted_psi'] = expit(df[f'{tissue}_logit_delta_pred'] + df['logit_mean_psi'])

            # === Compute Spearman correlation
            valid = (df[f'{tissue}_true'] >= 0) & (df[f'{tissue}_true'] <= 100) & (~df[f'{tissue}_true'].isnull())
            rho_psi, _ = spearmanr(df.loc[valid, f'{tissue}_true'], df.loc[valid, 'final_predicted_psi'])
            self.log("test_spearman_logit", rho_psi, prog_bar=True, sync_dist=True)
            print(f"\n🔬 Spearman ρ (PSI): {rho_psi:.4f}")

            # === Save to TSV
            df.to_csv(
                f"tsplice_final_predictions_{tissue.replace(' ', '_')}.tsv", sep="\t", index=False
            )
            print(df.head())

            ########################## differential psi splicing ##########################

            # ---- Merge on exon_id ----
            df = pd.merge(
                ground_truth[['exon_id', 'logit_mean_psi', tissue]],  # ground truth PSI
                predictions[['exon_id', tissue]],   # predicted logit(delta)
                on='exon_id',
                suffixes=('_true', '_logit_delta_pred')
            )


            # 1. Calculate ground-truth logit-delta for each exon
            eps = 1e-6
            df['Retina_frac_true'] = np.clip(df[f'{tissue}_true'] / 100, eps, 1 - eps)
            df['truth_delta_psi'] = logit(df['Retina_frac_true']) - df['logit_mean_psi']

            # 2. Prepare valid mask (avoid -1 and NaN in both columns)
            valid = (
                (df[f'{tissue}_true'] >= 0) & (df[f'{tissue}_true'] <= 100) & (~df[f'{tissue}_true'].isnull()) &
                (~df['logit_mean_psi'].isnull()) & (~df[f'{tissue}_logit_delta_pred'].isnull())
            )

            # 3. Compute Spearman correlation between truth_delta_psi and predicted logit-delta
            rho_delta, _ = spearmanr(
                df.loc[valid, 'truth_delta_psi'],
                df.loc[valid, f'{tissue}_logit_delta_pred']
            )
            self.log("test_spearman_Deltalogit", rho_delta, prog_bar=True, sync_dist=True)
            print(f"\n🔬 Spearman ρ (logit-delta): {rho_delta:.4f}")


        else:

            y_pred_all = torch.cat(self.test_preds).numpy()
            y_true_all = torch.cat(self.test_targets).numpy()

            # rho, _ = spearmanr(y_true_all, y_pred_all)
            # self.log("test_spearman", rho, prog_bar=True, sync_dist=True)
            # print(f"\n🔬 Spearman ρ (test set): {rho:.4f}")

            # Apply logit transformation with clamping to avoid log(0)
            eps = 1e-6
            y_true_logit = logit(np.clip(y_true_all/100, eps, 1 - eps))
            y_pred_logit = logit(np.clip(y_pred_all/100, eps, 1 - eps))

            rho, _ = spearmanr(y_true_logit, y_pred_logit)
            self.log("test_spearman_logit", rho, prog_bar=True, sync_dist=True)
            print(f"\n🔬 Spearman ρ (logit PSI, test set): {rho:.4f}")

            import time
            trimester = time.strftime("_%Y_%m_%d__%H_%M_%S")


            df = pd.DataFrame({
                "index": np.arange(len(y_true_all)),
                "y_true": y_true_all,
                "y_pred": y_pred_all,
                "y_true_logit": y_true_logit,
                "y_pred_logit": y_pred_logit,
            })
            df.to_csv(f"/mnt/home/at3836/Contrastive_Learning/files/test_predictions_with_index_{trimester}.csv", index=False)


    def on_train_epoch_start(self):
        self.epoch_start_time = time.time()

    def on_train_epoch_end(self):
        epoch_time = time.time() - self.epoch_start_time
        gpu_memory = torch.cuda.memory_allocated(0) / 1e9 if torch.cuda.is_available() else 0
        reserved_memory = torch.cuda.memory_reserved(0) / 1e9 if torch.cuda.is_available() else 0
        peak_memory = torch.cuda.max_memory_reserved(0) / 1e9 if torch.cuda.is_available() else 0

        self.log("epoch_time", epoch_time, prog_bar=True, sync_dist=True)
        self.log("gpu_memory_usage", gpu_memory, prog_bar=True, sync_dist=True)
        self.log("gpu_reserved_memory", reserved_memory, prog_bar=True, sync_dist=True)
        self.log("gpu_peak_memory", peak_memory, prog_bar=True, sync_dist=True)

        print(f"\nEpoch {self.current_epoch} took {epoch_time:.2f} seconds.")
        print(f"GPU Memory Used: {gpu_memory:.2f} GB, Reserved: {reserved_memory:.2f} GB, Peak: {peak_memory:.2f} GB")

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            print(f"Setting up training for {self.config.task._name_}")
        if stage == 'validate' or stage is None:
            print(f"Setting up validation for {self.config.task._name_}")

    def configure_optimizers(self):
        return instantiate(self.config.optimizer, params=self.parameters())
    



    