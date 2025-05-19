import sys
import os

base_dir = os.path.dirname(__file__)
ata_root = os.path.join(base_dir, "tisfm_original", "ATAConv-main")

if ata_root not in sys.path:
    sys.path.insert(0, ata_root)

# print("🛣 sys.path:", sys.path)
# Now you can import both the model and its dependencies
from models.model_pos_attention_calib_sigmoid_interaction import TISFM

