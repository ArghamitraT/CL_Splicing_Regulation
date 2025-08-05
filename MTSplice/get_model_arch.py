import os
from pkg_resources import resource_filename
# from tensorflow.keras.models import load_model
from tensorflow.keras.models import load_model
from mmsplice.layers import SplineWeight1D

# Define the relative model paths within the mmsplice package
_MODELS = ['models/mtsplice' + str(i) + '.h5' for i in range(8)]
_MODELS_DEEP = ['models/mtsplice_deep' + str(i) + '.h5' for i in range(4)]

# Use resource_filename to get full paths to bundled model files
MTSPLICE = [resource_filename('mmsplice', m) for m in _MODELS]
MTSPLICE_DEEP = [resource_filename('mmsplice', m) for m in _MODELS_DEEP]

# Inspect function
def inspect_model(path):
    print(f"\n🔍 Inspecting model: {path}")
    try:
        model = load_model(path, compile=False)
        model.summary()  # Show model architecture
    except Exception as e:
        print(f"❌ Error loading {path}: {e}")

# Inspect all models
print("\n--- MTSplice Models ---")
for path in MTSPLICE:
    inspect_model(path)

print("\n--- MTSplice Deep Models ---")
for path in MTSPLICE_DEEP:
    inspect_model(path)
