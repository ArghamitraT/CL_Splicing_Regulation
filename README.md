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
 ├── aux_models/
 ├── callbacks/
 ├── dataset/
 ├── embedder/
 ├── loss/
 ├── model/
 ├── optimizer/
 ├── task/
 ├── tokenizer/
 ├── trainer/
 ├── pretrain_CLADES.yaml
 └── finetune_CLADES.yaml

scripts/
 ├── pretrain_CLADES.sh
 ├── finetune_CLADES.sh
 ├── pretrain_CLADES.py
 └── finetune_CLADES.py
```
