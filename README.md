Sub A (2 cerințe) - Manipulare date + agregări |
Sub B (3 cerințe) - Metoda principală de analiză |
Sub C (1 cerință) - Calcul teoretic
----------------------------------------------------
A1 - Filtrare + Calcul medie/agregare
# 1. Citești CSV-uri
df = pd.read_csv("date.csv")

# 2. Identifici coloanele cu ani
years = [c for c in df.columns if c.isdigit()]
# SAU
years = [str(an) for an in range(2008, 2022)]

# 3. Calculezi medie/agregare pe rânduri
df["Medie"] = df[years].mean(axis=1)

# 4. Filtrare (diverse condiții)
rezultat = df[df["Medie"] > prag]

# 5. Selectezi coloane + sortare
rezultat = rezultat[["Col1", "Col2", "Medie"]].sort_values("Medie", ascending=False)

# 6. Salvezi
rezultat.to_csv("Cerinta1.csv", index=False)
------------------------------------------------------
A2 - Grupare pe categorii (județe/continente/etc)
# 1. Merge între dataframe-uri
df = df1.merge(df2[["Cheie", "Categorie"]], on="Cheie", how="left")

# 2. Grupare + agregare
rows = []
for categorie, grup in df.groupby("Categorie"):
    row = {"Categorie": categorie}
    
    # Calcul pe fiecare an/indicator
    for an in years:
        row[an] = grup[an].sum()  # sau .mean(), .max(), etc.
    
    rows.append(row)

# 3. DataFrame + salvare
rezultat = pd.DataFrame(rows).sort_values("Categorie")
rezultat.to_csv("Cerinta2.csv", index=False)

Ce variază la A2:
Tipul agregării: .sum(), .mean(), .max(), (grup[col] == 0).sum()
Criteriul de filtrare
Coloanele selectate
--------------------------------------------------------------------
SECȚIUNEA B - METODA PRINCIPALĂ
Aparițiile metodelor:

Clustering Ward - 3 din 6 subiecte 
PCA - 1 subiect 
Factor Analysis - 1 subiect 
LDA - 1 subiect 

Clustering Ward
Template fix (apare identic în 3 subiecte):
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 1. Extragem datele numerice
X = df[coloane_numerice].to_numpy(dtype=float)

# 2. STANDARDIZARE (OBLIGATORIU!)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Clustering Ward
Z = linkage(X_scaled, method="ward")

# B1) AFIȘARE MATRICE IERARHIE
print("Matrice ierarhie:")
print(Z)
# SAU salvare în CSV
pd.DataFrame(Z, columns=["C1", "C2", "Dist", "Nr"]).to_csv("h.csv", index=False)

# B2) DENDROGRAMA
plt.figure(figsize=(12, 6))
dendrogram(Z, no_labels=True)
plt.title("Dendrograma - metoda Ward")
plt.xlabel("Instante")
plt.ylabel("Distanta")
plt.tight_layout()
plt.show()

# B3) PARTIȚIA OPTIMĂ
k_opt = 3  # alegi din dendrogramă
clusters = fcluster(Z, t=k_opt, criterion="maxclust")

# Salvare partiție
popt = df[["ID", "Nume"]].copy()
popt["Cluster"] = clusters
popt.to_csv("popt.csv", index=False)
PCA - Template
pythonfrom sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 1. Standardizare
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# 2. PCA
pca = PCA()
scores = pca.fit_transform(X_std)

# B1) Varianțe (eigenvalues)
print(pca.explained_variance_)

# B2) Salvare scoruri
scores_df = pd.DataFrame(scores, columns=[f"PC{i+1}" for i in range(scores.shape[1])])
scores_df.to_csv("scoruri.csv", index=False)

# B3) Grafic PC1 vs PC2
plt.scatter(scores_df["PC1"], scores_df["PC2"])
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()
--------------------------------------------------
