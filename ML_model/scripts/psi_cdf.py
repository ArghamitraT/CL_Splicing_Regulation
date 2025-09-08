import pandas as pd
import os
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

tissue = "Retina___Eye"
data_dir = "/gpfs/commons/home/atalukder/Contrastive_Learning/data/ASCOT/psi_per_tissue/variable_cassette_exons/"
csv_path = os.path.join(data_dir, tissue) + "_psiWmean.csv"

df = pd.read_csv(csv_path)
psi = df["PSI"].values

res = stats.ecdf(psi)

ax = plt.subplot()
res.cdf.plot(ax)
ax.set_xlabel("PSI")
ax.set_ylabel("Empirical CDF")
plt.savefig("/gpfs/commons/home/nkeung/Contrastive_Learning/code/ML_model/scripts/figures/" + f"{tissue}_cdf.png", dpi=300, bbox_inches='tight')
