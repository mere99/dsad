#Structura examen DSAD 

- **Sectiunea A**: Manipulare date si agregari (2 cerinte)
- **Sectiunea B**: Metoda principala de analiza (3 cerinte)
- **Sectiunea C**: Calcul teoretic (1 cerinta)

---

## Sectiunea A - Manipulare Date si Agregari

### A1 - Filtrare si Calcul Medie/Agregare

```python
import pandas as pd

# 1. Citire CSV
df = pd.read_csv("date.csv")

# 2. Identificare coloane cu ani
years = [c for c in df.columns if c.isdigit()]
# SAU
years = [str(an) for an in range(2008, 2022)]

# 3. Calcul medie/agregare pe randuri
df["Medie"] = df[years].mean(axis=1)

# 4. Filtrare (diverse conditii)
prag = 100  # exemplu
rezultat = df[df["Medie"] > prag]

# 5. Selectare coloane si sortare
rezultat = rezultat[["Col1", "Col2", "Medie"]].sort_values("Medie", ascending=False)

# 6. Salvare rezultat
rezultat.to_csv("Cerinta1.csv", index=False)
```

### A2 - Grupare pe Categorii

```python
# 1. Merge intre dataframe-uri
df = df1.merge(df2[["Cheie", "Categorie"]], on="Cheie", how="left")

# 2. Grupare si agregare
rows = []
for categorie, grup in df.groupby("Categorie"):
    row = {"Categorie": categorie}
    
    # Calcul pe fiecare an/indicator
    for an in years:
        row[an] = grup[an].sum()  # sau .mean(), .max(), etc.
    
    rows.append(row)

# 3. Creare DataFrame si salvare
rezultat = pd.DataFrame(rows).sort_values("Categorie")
rezultat.to_csv("Cerinta2.csv", index=False)
```

**Variabile la A2:**
- Tipul agregarii: `.sum()`, `.mean()`, `.max()`, `(grup[col] == 0).sum()`
- Criteriul de filtrare
- Coloanele selectate

---

## Sectiunea B - Metoda Principala de Analiza

### Aparitiile Metodelor (din 6 subiecte)
- **Clustering Ward**: 3 subiecte
- **PCA**: 1 subiect
- **Factor Analysis**: 1 subiect
- **LDA**: 1 subiect

### B1 - Clustering Ward (Template Fix)

```python
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 1. Extragere date numerice
X = df[coloane_numerice].to_numpy(dtype=float)

# 2. STANDARDIZARE (OBLIGATORIU!)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Clustering Ward
Z = linkage(X_scaled, method="ward")

# B1.1) AFISARE MATRICE IERARHIE
print("Matrice ierarhie:")
print(Z)

# SAU salvare in CSV
pd.DataFrame(Z, columns=["C1", "C2", "Dist", "Nr"]).to_csv("h.csv", index=False)

# B1.2) DENDROGRAMA
plt.figure(figsize=(12, 6))
dendrogram(Z, no_labels=True)
plt.title("Dendrograma - metoda Ward")
plt.xlabel("Instante")
plt.ylabel("Distanta")
plt.tight_layout()
plt.show()

# B1.3) PARTITIA OPTIMA
k_opt = 3  # alegi din dendrograma
clusters = fcluster(Z, t=k_opt, criterion="maxclust")

# Salvare partitie
popt = df[["ID", "Nume"]].copy()
popt["Cluster"] = clusters
popt.to_csv("popt.csv", index=False)
```

### B2 - PCA (Principal Component Analysis)

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 1. Standardizare
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# 2. PCA
pca = PCA()
scores = pca.fit_transform(X_std)

# B2.1) Variante (eigenvalues)
print("Variante explicate:")
print(pca.explained_variance_)

# B2.2) Salvare scoruri
scores_df = pd.DataFrame(
    scores, 
    columns=[f"PC{i+1}" for i in range(scores.shape[1])]
)
scores_df.to_csv("scoruri.csv", index=False)

# B2.3) Grafic PC1 vs PC2
plt.figure(figsize=(10, 6))
plt.scatter(scores_df["PC1"], scores_df["PC2"])
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Scoruri PCA - PC1 vs PC2")
plt.grid(True)
plt.show()
```

---

## Sectiunea C - Calcul Teoretic

---

## Note Importante

- **Standardizarea** este obligatorie pentru Clustering Ward si PCA
- Coloanele numerice trebuie identificate corect inainte de analiza
- Valorile lipsa trebuie tratate inainte de procesare
- Numarul optim de clustere (k_opt) se determina din dendrograma

