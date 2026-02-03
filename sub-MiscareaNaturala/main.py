import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

import matplotlib.pyplot as plt


# =========================
# 0) Citire date
# =========================
mn = pd.read_csv("MiscareaNaturala.csv")
tari = pd.read_csv("CoduriTariExtins.csv")


# =========================
# A1) Tari cu RS sub media globala
# =========================
# Calculam media globala a ratei sporului natural
rs_medie = mn["RS"].mean()

# Selectam tarile cu RS mai mic decat media globala
cerinta1 = mn[mn["RS"] < rs_medie][
    ["Three_Letter_Country_Code", "Country_Name", "RS"]
]

# Sortam descrescator dupa RS
cerinta1 = cerinta1.sort_values("RS", ascending=False)

# Salvam rezultatul
cerinta1.to_csv("Cerinta1.csv", index=False)


# =========================
# A2) Pentru fiecare continent: tari cu valori maxime
# =========================
# Lipim continentul la fiecare tara
df = mn.merge(
    tari[["Three_Letter_Country_Code", "Continent_Name"]],
    on="Three_Letter_Country_Code",
    how="left"
)

indicators = ["FR", "IM", "LE", "LEF", "LEM", "MMR", "RS"]

rows = []
for cont, g in df.groupby("Continent_Name"):
    row = {"Continent_Name": cont}
    for ind in indicators:
        # Tara cu valoarea maxima pentru indicator
        idx = g[ind].idxmax()
        row[ind] = g.loc[idx, "Three_Letter_Country_Code"]
    rows.append(row)

cerinta2 = pd.DataFrame(rows)

# Salvam rezultatul
cerinta2.to_csv("Cerinta2.csv", index=False)


# =========================
# B) Clusterizare Ward
# =========================
# Selectam doar indicatorii numerici
X = mn[indicators].to_numpy(dtype=float)

# Standardizare
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# Clusterizare Ward
Z = linkage(X_std, method="ward")

# B1) Salvam matricea ierarhiei
h = pd.DataFrame(
    Z,
    columns=["Cluster1", "Cluster2", "Distance", "Nr_instances"]
)
h.to_csv("h.csv", index=False)

# B2) Dendrograma
plt.figure(figsize=(12, 6))
dendrogram(Z, no_labels=True)
plt.title("Dendrograma - metoda Ward")
plt.xlabel("Instante (tari)")
plt.ylabel("Distanta")
plt.tight_layout()
plt.show()

# B3) Partitia optima
# Alegem k din dendrograma (valoare implicita)
k_opt = 3

clusters = fcluster(Z, t=k_opt, criterion="maxclust")

popt = mn[["Three_Letter_Country_Code", "Country_Name"]].copy()
popt["Cluster"] = clusters

popt.to_csv("popt.csv", index=False)
