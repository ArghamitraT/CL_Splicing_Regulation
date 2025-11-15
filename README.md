# CLADES
Contrastive Learning Augmented DifferEntial Splicing with Orthologous Positive Pairs

### 1. Installation

Clone the repository and create the CLADES environment:

```bash
git clone https://github.com/ArghamitraT/CLADES.git
cd CLADES

conda env create -n clades_env -f environment.yml
conda activate clades_env

pip install -e .
```

### 2. Set up Weights & Biases (W&B)
Before running any training script, edit the following Hydra config files:

- `configs/pretrain_CLADES.yaml`
- `configs/finetune_CLADES.yaml`

```yaml
wandb:
  api_key: "NEEDED"
```
and the replacement:

```yaml
wandb:
  api_key: "<YOUR_WANDB_API_KEY>"
```

### 3. Pre-training

- Option A — Bash 
```bash
cd scripts
bash pretrain_CLADES.sh
```

- Option B — Python
```bash
cd scripts
python pretrain_CLADES.py
```
The following Hydra overrides are commonly adjusted during CLADES contrastive pretraining:

- **task=introns_cl** — activates CLADES contrastive learning pipeline  
- **embedder="mtsplice"** — MTSplice-style dual-branch encoder  
- **tokenizer="onehot_tokenizer"** — one-hot intron boundary tokenization  
- **loss="supcon"** — supervised contrastive loss  
- **dataset.n_augmentations** — number of positive species views (default = 2)  
- **trainer.max_epochs** — total pretraining epochs  
- **task.global_batch_size** — global batch size across GPUs  
- **optimizer.lr** — contrastive learning learning rate  
- **dataset.min_views** — required number of species views per exon (default ≥ 30)  
- **dataset.fivep_ovrhang**, **dataset.threep_ovrhang** — intron window sizes  

For full details, see: `scripts/pretrain_CLADES.sh`


### 4. Fine-tuning

- Option A — Bash 
```bash
cd scripts
bash finetune_CLADES.sh
```

- Option B — Python
```bash
cd scripts
python finetune_CLADES.py
```

Important configurable parameters:

- **aux_models.mtsplice_weights** — path to pretrained CLADES encoder
- **aux_models.eval_weights** — checkpoint for evaluation-only mode
- **aux_models.freeze_encoder** — freeze encoder (true) or fine-tune (false)
- **aux_models.warm_start** — initialize from pretrained encoder
- **optimizer.lr** — learning rate
- **trainer.max_epochs** — number of epochs
- **dataset.fivep_ovrhang, dataset.threep_ovrhang** — intron window sizes

For full details, see: `scripts/finetune_CLADES.sh`

### 📂 Sample data

`data/` folder contains sample data for pre-training and finetuning


### 📂 Output Organization

All training runs create timestamped directories under `output/`, for example:

```bash
output/
├── pretrain_2025_11_14_23_12_22/
└── finetune_2025_11_14_23_46_21/
```

Each run contains:

```bash
output/<run_name>/
├── hydra/ # Hydra config snapshots
├── wandb/ # Weights & Biases logs
└── checkpoints/ # Model checkpoints
```

### 🗂️ Configuration Layout
```bash
configs/
 ├── aux_models/       # Pretrained model weights, MTSplice settings, eval model paths
 ├── callbacks/        # Lightning callbacks (checkpointing, early stopping, LR schedulers)
 ├── dataset/          # Dataset parameters (paths, window sizes, species settings)
 ├── embedder/         # Encoder architecture configs (MTSplice, ResNet, TISFM, etc.)
 ├── loss/             # Loss function configs (SupCon, NT-Xent, BCE, KL)
 ├── model/            # Full model assembly (encoder + projection head)
 ├── optimizer/        # Optimizer and scheduler settings
 ├── task/             # Task definitions (contrastive training, PSI regression)
 ├── tokenizer/        # Sequence tokenization settings (one-hot, k-mer, etc.)
 ├── trainer/          # Lightning Trainer config (devices, precision, epochs)
 ├── pretrain_CLADES.yaml    # Main config for contrastive pretraining
 └── finetune_CLADES.yaml    # Main config for PSI regression fine-tuning


scripts/
 ├── pretrain_CLADES.sh   # SLURM/bash wrapper for contrastive pretraining
 ├── finetune_CLADES.sh   # SLURM/bash wrapper for PSI regression fine-tuning
 ├── pretrain_CLADES.py   # Python entry point for contrastive pretraining
 └── finetune_CLADES.py   # Python entry point for PSI regression fine-tuning

 src/
 ├── datasets/          # Dataset classes for contrastive pretraining & PSI regression
 │    ├── base.py               # Abstract dataset + shared utilities
 │    ├── introns_alignment.py  # Loads cross-species intron alignment data
 │    ├── auxiliary_jobs.py     # Helper preprocessing utilities
 │    └── lit.py                # LightningDataModule
 │
 ├── embedder/          # Encoders (MTSplice, ResNet-style, TISFM, etc.)
 │    ├── base.py
 │    ├── utils.py
 │    └── mtsplice/
 │
 ├── loss/              # Loss functions (contrastive + regression)
 │    ├── MTSPLiceBCELoss.py
 │    └── supcon.py
 │
 ├── model/             # LightningModule model definitions
 │    ├── lit.py
 │    ├── simclr.py
 │    └── MTSpliceBCE.py
 │
 ├── tokenizers/        # Sequence tokenizers
 │    └── onehot_tokenizer.py
 │
 ├── trainer/           # Trainer-level utilities
 │    └── utils.py
 │
 └── utils/             # Global project utilities
      ├── config.py
      └── utils.py

```
