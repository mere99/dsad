import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

import matplotlib.pyplot as plt

# =========================
# 0) Citire date
# =========================
# Citim cele 3 fisiere CSV (trebuie sa fie in acelasi folder cu scriptul)
div = pd.read_csv("Diversitate.csv")
cod = pd.read_csv("Coduri_Localitati.csv")
g20 = pd.read_csv("g20.csv")

# Identificam automat coloanele care sunt ani: "2008", "2009", ..., "2021"
years = [c for c in div.columns if c.isdigit()]


# =========================
# A1) Media indicelui pe localitate
# =========================
# Calculam media pe ani pentru fiecare localitate
div["Diversitate"] = div[years].mean(axis=1)

# Selectam coloanele cerute si sortam descrescator dupa media diversitatii
cerinta1 = (
    div[["Siruta", "Localitate", "Diversitate"]]
    .sort_values("Diversitate", ascending=False)
)

# Salvam rezultatul in Cerinta1.csv
cerinta1.to_csv("Cerinta1.csv", index=False)


# =========================
# A2) Numar localitati cu diversitate 0 pe judet si an
# =========================
# Lipim (merge) judetul la fiecare localitate folosind Siruta ca cheie
merged = div.merge(cod[["Siruta", "Judet"]], on="Siruta", how="left")

# Pentru fiecare judet, numaram cate localitati au valoarea 0 in fiecare an
rows = []
for judet, g in merged.groupby("Judet"):
    row = {"Judet": judet}
    for y in years:
        row[y] = (g[y] == 0).sum()
    rows.append(row)

cerinta2 = pd.DataFrame(rows).sort_values("Judet")

# Salvam rezultatul in Cerinta2.csv
cerinta2.to_csv("Cerinta2.csv", index=False)


# =========================
# B) Clusterizare Ward pe indicii de diversitate
# =========================
# Luam doar valorile pe ani ca matrice (rand = localitate, coloana = an)
X = div[years].to_numpy(dtype=float)

# Standardizare: aduce datele la aceeasi scara (medie 0, deviatie standard 1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Aplicam clusterizare ierarhica prin metoda Ward
# Z este matricea ierarhiei (jonctiuni, distante, dimensiuni cluster)
Z = linkage(X_scaled, method="ward")

# B1) Afisam matricea ierarhiei la consola
print("Matrice ierarhie (Ward) - format: [c1, c2, distanta, nr_instante]")
print(Z)

# B2) Afisam dendrograma
plt.figure(figsize=(12, 6))
dendrogram(Z, no_labels=True)
plt.title("Dendrograma - metoda Ward")
plt.xlabel("Instante (localitati)")
plt.ylabel("Distanta")
plt.tight_layout()
plt.show()

# B3) Partitia "optima"
# La examen, alegi numarul de clustere (k) dupa dendrograma.
# Aici punem o valoare implicita, o poti schimba usor.
k_opt = 3

# Fiecare localitate primeste un numar de cluster: 1..k_opt
clusters = fcluster(Z, t=k_opt, criterion="maxclust")

# Salvam partitia in popt.csv
popt = div[["Siruta", "Localitate"]].copy()
popt["Cluster"] = clusters
popt.to_csv("popt.csv", index=False)


# =========================
# C) Factorul specific cu varianta maxima (1..5)
# =========================
# Avem 5 variabile observate si 2 factori comuni.
# In g20.csv sunt loadings (L).
# Communality pentru variabila i: h_i^2 = sum_j (L_ij^2)
# Varianta specifica pentru variabila i: psi_i = 1 - h_i^2
# Cerinta: afisam indexul (1..5) cu psi_i maxim.

# Luam doar coloanele numerice (loadings)
L = g20.select_dtypes(include=[np.number]).to_numpy(dtype=float)

communality = (L ** 2).sum(axis=1)
specific_var = 1 - communality

idx_max = int(np.argmax(specific_var)) + 1
print("Factorul specific cu varianta maxima (1..5):", idx_max)
