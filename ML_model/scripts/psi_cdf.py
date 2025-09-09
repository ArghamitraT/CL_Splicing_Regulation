import pandas as pd
import os
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

tissue = "Lung"
data_dir = "/gpfs/commons/home/atalukder/Contrastive_Learning/data/ASCOT/psi_per_tissue/variable_cassette_exons/"
csv_path = os.path.join(data_dir, tissue) + "_psiWmean.csv"

df = pd.read_csv(csv_path)
psi = df["PSI"].values

res = stats.ecdf(psi)

# Calculate proportion of exons in low or high
p_low = res.cdf.evaluate(50)
p_high = 1 - p_low

print(f"\nTissue: {tissue}\n")
print(f"Low: {p_low}")
print(f"High: {p_high}")

ax = plt.subplot()
res.cdf.plot(ax)
ax.set_xlabel("PSI")
ax.set_ylabel("Empirical CDF")
ax.set_title(f"{tissue} PSI ECDF")
plt.savefig("/gpfs/commons/home/nkeung/Contrastive_Learning/code/ML_model/scripts/figures/" + f"{tissue}_cdf.png", dpi=300, bbox_inches='tight')
