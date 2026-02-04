import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt

print("=" * 70)
print("REZOLVARE SUBIECT 6 - INDICATORI MACROECONOMICI")
print("=" * 70)

# ============================================================================
# CITIRE DATE
# ============================================================================

print("\n=== PASUL 0: CITIRE FISIERE ===\n")

# Citim fisierul cu indicatorii macroeconomici
df_indicatori = pd.read_csv('GlobalIndicatorsPerCapita_2021.csv')

# Citim fisierul cu codurile tarilor
df_coduri = pd.read_csv('CoduriTari.csv')

print("Primele randuri din GlobalIndicatorsPerCapita_2021.csv:")
print(df_indicatori.head())
print(f"\nNumar tari: {len(df_indicatori)}")
print()

print("Coloanele disponibile:")
print(df_indicatori.columns.tolist())
print()

print("Primele randuri din CoduriTari.csv:")
print(df_coduri.head())
print()

# ============================================================================
# CERINTA A.1 - RAMURA CU VALOAREA ADAUGATA CEA MAI MARE
# ============================================================================

print("\n=== CERINTA A.1: RAMURA DOMINANTA PE TARA ===\n")

# Identificam coloanele cu ramurile economice
# Conform enuntului: de la AgrHuntForFish la Other
ramuri = ['AgrHuntForFish', 'Construction', 'Manufacturing',
          'MiningManUt', 'TradeT', 'TransportComm', 'Other']

print(f"Ramuri economice: {ramuri}\n")

# Cream o lista pentru a stoca rezultatele
rezultate_a1 = []

# Pentru fiecare tara (fiecare rand)
for idx, rand in df_indicatori.iterrows():
    # Extragem valorile pentru toate ramurile
    valori_ramuri = rand[ramuri]

    # Gasim ramura cu valoarea maxima
    # idxmax() returneaza numele coloanei cu valoarea maxima
    ramura_dominanta = valori_ramuri.idxmax()

    # Extragem CountryID si Country
    country_id = rand['CountryID']
    country_name = rand['Country']

    # Adaugam in lista
    rezultate_a1.append({
        'CountryID': country_id,
        'Country': country_name,
        'Ramura_Dominanta': ramura_dominanta
    })

# Cream DataFrame din rezultate
df_cerinta1 = pd.DataFrame(rezultate_a1)

print("Rezultate (primele 10 tari):")
print(df_cerinta1.head(10))
print()

# SALVARE
df_cerinta1.to_csv('Cerinta1.csv', index=False)

print(">>> Fisierul Cerinta1.csv a fost salvat!\n")

# Statistici
print("Distributie ramuri dominante:")
print(df_cerinta1['Ramura_Dominanta'].value_counts())
print()

# ============================================================================
# CERINTA A.2 - TARI CU VALORI MAXIME PE CONTINENT
# ============================================================================

print("\n=== CERINTA A.2: TARI CU VALORI MAXIME PE CONTINENT ===\n")

# Unim datele cu informatiile despre continente
df_complet = df_indicatori.merge(df_coduri[['CountryID', 'Continent']],
                                 on='CountryID',
                                 how='left')

print("Date dupa unire (primele 3 randuri):")
print(df_complet[['Country', 'CountryID', 'Continent', 'GNI']].head(3))
print()

# Identificam toti indicatorii (de la GNI pana la sfarsit)
# Conform enuntului: de la GNI pana la Other
indicatori_economici = ['GNI', 'ChangesInv', 'Exports', 'Imports',
                        'FinalConsExp', 'GrossCF', 'HouseholdConsExp',
                        'AgrHuntForFish', 'Construction', 'Manufacturing',
                        'MiningManUt', 'TradeT', 'TransportComm', 'Other']

print(f"Indicatori economici: {indicatori_economici}\n")

# Continentele gasite
continente = df_complet['Continent'].dropna().unique()
print(f"Continente gasite: {continente}\n")

# Cream o lista pentru a stoca rezultatele
rezultate_a2 = []

# Pentru fiecare continent
for continent in continente:
    # Filtram datele pentru continentul curent
    df_continent = df_complet[df_complet['Continent'] == continent]

    # Cream un dictionar pentru acest continent
    row = {'Continent': continent}

    # Pentru fiecare indicator, gasim tara cu valoarea maxima
    for indicator in indicatori_economici:
        # Gasim indexul cu valoarea maxima pentru acest indicator
        idx_max = df_continent[indicator].idxmax()

        # Extragem CountryID-ul tarii cu valoarea maxima
        country_id_max = df_continent.loc[idx_max, 'CountryID']

        # Salvam in dictionar
        row[indicator] = country_id_max

    # Adaugam in lista
    rezultate_a2.append(row)

# Cream DataFrame din rezultate
df_cerinta2 = pd.DataFrame(rezultate_a2)

print("Rezultate:")
print(df_cerinta2)
print()

# SALVARE
df_cerinta2.to_csv('Cerinta2.csv', index=False)

print(">>> Fisierul Cerinta2.csv a fost salvat!\n")

# ============================================================================
# CERINTA B - ANALIZA DE CLUSTERI (METODA WARD)
# ============================================================================

print("\n=== CERINTA B: ANALIZA DE CLUSTERI ===\n")

# Extragem doar coloanele cu indicatorii macroeconomici
X = df_indicatori[indicatori_economici].copy()

print(f"Dimensiune date pentru clustering: {X.shape}")
print("(tari x indicatori)\n")

# Eliminam randurile cu valori lipsa (daca exista)
X_clean = X.dropna()
print(f"Dimensiune dupa eliminare valori lipsa: {X_clean.shape}\n")

# Salvam indexurile (pentru a sti care tari au ramas)
indici_valide = X_clean.index

# STANDARDIZARE (obligatorie pentru clustering)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_clean)

print("Date standardizate (primele 3 randuri):")
print(X_scaled[:3])
print()

# CLUSTERIZARE WARD
Z = linkage(X_scaled, method='ward')

print("Clusterizare Ward aplicata cu succes!\n")
print(f"Dimensiune matrice ierarhie: {Z.shape}")
print("(jonctiuni x 4 coloane: cluster1, cluster2, distanta, nr_instante)\n")

# ============================================================================
# CERINTA B.1 - DENDROGRAMA CU 3 CLUSTERI (CORECTAT)
# ============================================================================

print("\n=== CERINTA B.1: DENDROGRAMA ===\n")

# Cream partitia cu 3 clusteri
k = 3
clusters = fcluster(Z, t=k, criterion='maxclust')

print(f"Numar de clusteri: {k}")
print(f"Distributie tari pe clusteri:")
for i in range(1, k + 1):
    nr_tari = np.sum(clusters == i)
    print(f"  Cluster {i}: {nr_tari} tari")
print()

# Calculam distanta de taiere pentru k clusteri
# Distanta de taiere = mijlocul dintre ultimele 2 jonctiuni relevante
if len(Z) >= k:
    dist_k_minus_1 = Z[-(k - 1), 2]  # distanta pentru k-1 clusteri
    dist_k = Z[-k, 2] if k < len(Z) else 0  # distanta pentru k clusteri

    # Taiem la mijloc
    distanta_taiere = (dist_k_minus_1 + dist_k) / 2

    print(f"Distanta pentru {k} clusteri: {dist_k:.2f}")
    print(f"Distanta pentru {k - 1} clusteri: {dist_k_minus_1:.2f}")
    print(f"Distanta de taiere (mijloc): {distanta_taiere:.2f}\n")
else:
    distanta_taiere = None

# Cream figura pentru dendrograma
plt.figure(figsize=(14, 8))

# Plotam dendrograma fara etichete (pentru claritate)
dendrogram(Z,
           no_labels=True,
           color_threshold=distanta_taiere,
           above_threshold_color='gray')

# Adaugam linia de taiere
if distanta_taiere:
    plt.axhline(y=distanta_taiere, color='red', linestyle='--',
                linewidth=3, label=f'Taiere pentru {k} clusteri')
    plt.legend(fontsize=12, loc='upper right')

plt.title(f'Dendrograma - Metoda Ward ({k} clusteri)',
          fontsize=16, fontweight='bold')
plt.xlabel('Tari (instante)', fontsize=14)
plt.ylabel('Distanta', fontsize=14)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()

plt.savefig('dendrograma_3clusteri.png', dpi=300, bbox_inches='tight')

print(">>> Graficul dendrograma_3clusteri.png a fost salvat!\n")

plt.show()

# ============================================================================
# CERINTA B.2 - PARTITIA CU 3 CLUSTERI + SCOR SILHOUETTE
# ============================================================================

print("\n=== CERINTA B.2: PARTITIA CU SCORURI SILHOUETTE ===\n")

# Calculam scorurile Silhouette pentru fiecare instanta
silhouette_vals = silhouette_samples(X_scaled, clusters)

# Calculam scorul Silhouette global (media)
silhouette_avg = silhouette_score(X_scaled, clusters)

print(f"Scor Silhouette mediu: {silhouette_avg:.4f}")
print()

print("Interpretare Silhouette:")
print("  > 0.7: Structura puternica")
print("  > 0.5: Structura rezonabila")
print("  > 0.25: Structura slaba")
print("  < 0.25: Fara structura substantiala")
print()

# Cream DataFrame cu partitia
df_partitie = pd.DataFrame({
    'CountryID': df_indicatori.loc[indici_valide, 'CountryID'].values,
    'Country': df_indicatori.loc[indici_valide, 'Country'].values,
    'Cluster': clusters,
    'Silhouette': silhouette_vals
})

print("Partitie (primele 10 tari):")
print(df_partitie.head(10))
print()

# Statistici per cluster
print("Statistici Silhouette per cluster:")
for i in range(1, k + 1):
    vals = df_partitie[df_partitie['Cluster'] == i]['Silhouette']
    print(f"  Cluster {i}: medie = {vals.mean():.4f}, "
          f"min = {vals.min():.4f}, max = {vals.max():.4f}")
print()

# SALVARE
df_partitie.to_csv('p3.csv', index=False)

print(">>> Fisierul p3.csv a fost salvat!\n")

# ============================================================================
# CERINTA B.3 - PLOT PARTITIE IN AXE PRINCIPALE
# ============================================================================

print("\n=== CERINTA B.3: PLOT PARTITIE IN AXE PRINCIPALE ===\n")

# Aplicam PCA pentru a reduce la 2 dimensiuni (pentru vizualizare)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print(f"Varianta explicata de primele 2 componente: "
      f"{pca.explained_variance_ratio_.sum() * 100:.2f}%\n")

# Cream figura
plt.figure(figsize=(12, 8))

# Definim culori pentru fiecare cluster
culori = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
markeri = ['o', 's', '^', 'D', 'v', 'p']

# Plotam fiecare cluster separat
for i in range(1, k + 1):
    # Filtram datele pentru clusterul curent
    mask = clusters == i

    # Plotam punctele
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1],
                c=culori[i - 1],
                marker=markeri[i - 1],
                label=f'Cluster {i}',
                alpha=0.6,
                s=100,
                edgecolors='black',
                linewidth=0.5)

# Adaugam titlu si etichete
plt.title('Plot Partitie in Axe Principale (PCA)',
          fontsize=14, fontweight='bold')
plt.xlabel(f'Prima Componenta Principala (PC1) - '
           f'{pca.explained_variance_ratio_[0] * 100:.1f}% varianta',
           fontsize=12)
plt.ylabel(f'A Doua Componenta Principala (PC2) - '
           f'{pca.explained_variance_ratio_[1] * 100:.1f}% varianta',
           fontsize=12)

# Adaugam legenda
plt.legend(fontsize=10, loc='best')

# Adaugam grila
plt.grid(True, alpha=0.3)

# Adaugam linii pentru axe
plt.axhline(y=0, color='k', linewidth=0.5)
plt.axvline(x=0, color='k', linewidth=0.5)

# Salvam graficul
plt.tight_layout()
plt.savefig('plot_partitie_axe_principale.png', dpi=300, bbox_inches='tight')

print(">>> Graficul plot_partitie_axe_principale.png a fost salvat!\n")

plt.show()

# ============================================================================
# BONUS - GRAFIC SILHOUETTE
# ============================================================================

print("\n=== BONUS: GRAFIC SILHOUETTE ===\n")

fig, ax = plt.subplots(figsize=(10, 8))

y_lower = 10

# Pentru fiecare cluster
for i in range(1, k + 1):
    # Extragem scorurile Silhouette pentru clusterul curent
    cluster_silhouette_vals = silhouette_vals[clusters == i]

    # Sortam
    cluster_silhouette_vals.sort()

    size_cluster_i = cluster_silhouette_vals.shape[0]
    y_upper = y_lower + size_cluster_i

    # Plotam
    ax.fill_betweenx(np.arange(y_lower, y_upper),
                     0, cluster_silhouette_vals,
                     facecolor=culori[i - 1],
                     edgecolor=culori[i - 1],
                     alpha=0.7,
                     label=f'Cluster {i}')

    # Adaugam eticheta cu numarul clusterului
    ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

    y_lower = y_upper + 10

# Linia pentru scorul mediu
ax.axvline(x=silhouette_avg, color="red", linestyle="--",
           linewidth=2, label=f'Scor mediu ({silhouette_avg:.3f})')

ax.set_title("Grafic Silhouette pentru 3 Clusteri",
             fontsize=14, fontweight='bold')
ax.set_xlabel("Coeficient Silhouette", fontsize=12)
ax.set_ylabel("Cluster", fontsize=12)
ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig('grafic_silhouette.png', dpi=300, bbox_inches='tight')

print(">>> Graficul grafic_silhouette.png a fost salvat!\n")

plt.show()

# ============================================================================
# CERINTA C - CALCULARE SCORURI PENTRU INSTANTA NOUA
# ============================================================================

print("\n=== CERINTA C: SCORURI PENTRU INSTANTA NOUA ===\n")

# Citim matricea vectorilor proprii
df_vectori = pd.read_csv('a.csv')

print("Matricea vectorilor proprii:")
print(df_vectori)
print()

# Extragem matricea (randuri = variabile, coloane = componente)
vectori_proprii = df_vectori.values

print(f"Dimensiune matrice: {vectori_proprii.shape}")
print("(5 variabile x 5 componente)\n")

# Valorile proprii (eigenvalues) date in enunt
valori_proprii = np.array([3.24, 1.21, 0.36, 0.16, 0.01])

print("Valori proprii (eigenvalues):")
print(valori_proprii)
print()

# Instanta pentru care calculam scorurile
x = np.array([3, 1, 2, 1, 4])

print("Instanta x:")
print(x)
print()

# CALCULARE SCORURI
# Formula: scor_i = suma(x_j * vector_propriu_ji)
# unde j = variabile, i = componente

# SAU mai simplu: scoruri = x · matricea_vectorilor_proprii
scoruri = np.dot(x, vectori_proprii)

print("CALCULARE PAS CU PAS:\n")

# Afisam calculul pentru fiecare componenta
for i in range(len(scoruri)):
    print(f"Scor Componenta {i + 1} (PC{i + 1}):")
    print(f"  = {x[0]} * {vectori_proprii[0, i]:.4f} + "
          f"{x[1]} * {vectori_proprii[1, i]:.4f} + "
          f"{x[2]} * {vectori_proprii[2, i]:.4f} + "
          f"{x[3]} * {vectori_proprii[3, i]:.4f} + "
          f"{x[4]} * {vectori_proprii[4, i]:.4f}")
    print(f"  = {scoruri[i]:.4f}\n")

print("=" * 60)
print("REZULTAT FINAL - SCORURI PENTRU INSTANTA x:")
print("=" * 60)

for i in range(len(scoruri)):
    print(f"PC{i + 1}: {scoruri[i]:.4f}")

print("=" * 60)
print()

# Verificare: calculam si varianta explicata
varianta_totala = np.sum(valori_proprii)
procente_varianta = (valori_proprii / varianta_totala) * 100

print("INFORMATII SUPLIMENTARE:\n")
print("Varianta explicata de fiecare componenta:")
for i in range(len(valori_proprii)):
    print(f"  PC{i + 1}: {procente_varianta[i]:.2f}%")
print()

print(f"Varianta totala explicata de primele 2 componente: "
      f"{procente_varianta[:2].sum():.2f}%")
print()
