import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt


# =========================
# 0) Citire date
# =========================
aer = pd.read_csv("CalitateaAeruluiTari.csv")
tari = pd.read_csv("CoduriTari.csv")
print(aer.columns)
print(tari.columns)


# Indicatorii ceruti in subiect (coloanele cu masuratori)
indicators = [
    "Air_quality_Carbon_Monoxide",
    "Air_quality_Ozone",
    "Air_quality_Nitrogen_dioxide",
    "Air_quality_Sulphur_dioxide",
    "Air_quality_PM2.5",
    "Air_quality_PM10",
]


# =========================
# A1) Coeficient de variatie (CV) pe fiecare indicator
# =========================
# CV = standard_deviation / mean
# Folosim std cu ddof=1 (standard in statistica pe esantion)

rows = []
for col in indicators:
    mean_val = aer[col].mean()
    std_val = aer[col].std(ddof=1)

    # Evitam impartirea la 0 (daca media ar fi 0)
    cv = np.nan
    if mean_val != 0:
        cv = std_val / mean_val

    rows.append({"Indicator": col, "CV": cv})

cerinta1 = pd.DataFrame(rows)
cerinta1.to_csv("Cerinta1.csv", index=False)


# =========================
# A2) Pentru fiecare continent: indicatorul cu CV maxim
# =========================
# Lipim continentul la fiecare tara folosind CountryId
df = aer.merge(tari[["CountryID", "Continent"]],
               left_on="CountryId",
               right_on="CountryID",
               how="left")


rows2 = []
for cont, g in df.groupby("Continent"):
    best_indicator = None
    best_cv = -np.inf

    for col in indicators:
        mean_val = g[col].mean()
        std_val = g[col].std(ddof=1)

        if mean_val == 0:
            cv = np.nan
        else:
            cv = std_val / mean_val

        # Alegem valoarea maxima (ignorand NaN)
        if pd.notna(cv) and cv > best_cv:
            best_cv = cv
            best_indicator = col

    rows2.append({"Continent": cont, "Indicator": best_indicator, "CV": best_cv})

cerinta2 = pd.DataFrame(rows2).sort_values("Continent")
cerinta2.to_csv("Cerinta2.csv", index=False)


# =========================
# B) PCA standardizata pe indicatori
# =========================
# 1) Standardizare
X = df[indicators].to_numpy(dtype=float)
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# 2) PCA
pca = PCA()
scores = pca.fit_transform(X_std)

# B1) Variantele componentelor (eigenvalues)
print("Variances of principal components:")
print(pca.explained_variance_)

# B2) Salvam scorurile in scoruri.csv
# Salvam CountryId, Country si scorurile pe toate componentele
score_cols = [f"PC{i+1}" for i in range(scores.shape[1])]
scores_df = pd.DataFrame(scores, columns=score_cols)

out = pd.concat([df[["CountryId", "Country"]].reset_index(drop=True), scores_df], axis=1)
out.to_csv("scoruri.csv", index=False)

# B3) Grafic scoruri pe primele doua componente (PC1 vs PC2)
plt.figure(figsize=(10, 6))
plt.scatter(out["PC1"], out["PC2"])
plt.title("Scores plot: PC1 vs PC2")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.show()


# =========================
# C) LDA - numar minim de variabile discriminante > 90%
# =========================
# Avem valorile proprii (eigenvalues) date in cerinta
eigs = [0.9, 0.05, 0.8, 0.75, 0.5, 0.3, 0.2]

# Sortam descrescator (puterea cea mai mare prima)
eigs_sorted = sorted(eigs, reverse=True)

total = sum(eigs_sorted)
cum = 0.0
k = 0

for val in eigs_sorted:
    cum += val
    k += 1
    if cum / total > 0.90:
        break

print("Minimum number of discriminant variables for >90%:", k)
