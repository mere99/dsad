import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
from factor_analyzer import calculate_kmo

# Setam encoding pentru caracterele romanesti
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("=" * 70)
print("REZOLVARE SUBIECT 4 - CALITATEA AERULUI")
print("=" * 70)

# ============================================================================
# CERINTA A.1 - Tara cu valoarea maxima pentru fiecare indicator
# ============================================================================

print("\n=== CERINTA A.1 ===\n")

# Citim fisierul cu datele despre calitatea aerului
df_calitate = pd.read_csv('CalitateaAeruluiTari.csv')

print("Primele randuri din CalitateaAeruluiTari.csv:")
print(df_calitate.head())
print()

# Identificam coloanele cu indicatorii de calitate a aerului
# Toate coloanele care incep cu "Air_quality_"
indicatori = [col for col in df_calitate.columns if col.startswith('Air_quality_')]

print(f"Indicatori gasiti: {indicatori}\n")

# Cream o lista pentru a stoca rezultatele
rezultate_a1 = []

# Pentru fiecare indicator, gasim tara cu valoarea maxima
for indicator in indicatori:
    # Gasim indexul randului cu valoarea maxima pentru acest indicator
    idx_max = df_calitate[indicator].idxmax()

    # Extragem tara corespunzatoare
    tara_max = df_calitate.loc[idx_max, 'Country']

    # Adaugam in lista de rezultate
    rezultate_a1.append({
        'Indicator': indicator,
        'Country': tara_max
    })

    print(f"{indicator}: {tara_max}")

# Cream DataFrame din rezultate
df_cerinta1 = pd.DataFrame(rezultate_a1)

# Salvam in fisier CSV
df_cerinta1.to_csv('Cerinta1.csv', index=False)

print("\n>>> Fisierul Cerinta1.csv a fost salvat!")

# ============================================================================
# CERINTA A.2 - Tarile cu valori maxime pe continent pentru fiecare indicator
# ============================================================================

print("\n\n=== CERINTA A.2 ===\n")

# Citim fisierul cu codurile tarilor si continentele
df_coduri = pd.read_csv('CoduriTari.csv')

print("Primele randuri din CoduriTari.csv:")
print(df_coduri.head())
print()

# Unim datele de calitate cu informatiile despre continente
# Facem JOIN pe coloana CountryId
df_complet = pd.merge(df_calitate, df_coduri[['CountryId', 'Continent']],
                      on='CountryId', how='left')

print("Date dupa unire:")
print(df_complet.head())
print()

# Grupam datele pe continente
continente = df_complet['Continent'].unique()
print(f"Continente gasite: {continente}\n")

# Cream un dictionar pentru a stoca tarile cu valori maxime pe fiecare continent
rezultate_continent = {}

# Pentru fiecare continent
for continent in continente:
    # Filtram datele pentru continentul curent
    df_continent = df_complet[df_complet['Continent'] == continent]

    # Pentru fiecare indicator, gasim tara cu valoarea maxima
    tari_max = []

    for indicator in indicatori:
        # Gasim indexul cu valoarea maxima pentru acest indicator in continentul curent
        idx_max = df_continent[indicator].idxmax()

        # Extragem tara
        tara_max = df_continent.loc[idx_max, 'Country']
        tari_max.append(tara_max)

    # Salvam rezultatul pentru acest continent
    rezultate_continent[continent] = tari_max

# Cream DataFrame din rezultate
# Continentul va fi prima coloana, apoi fiecare indicator ca o coloana
df_cerinta2 = pd.DataFrame(rezultate_continent).T  # .T = transpunem
df_cerinta2.columns = indicatori
df_cerinta2.index.name = 'Continent'
df_cerinta2.reset_index(inplace=True)

print("Rezultate pe continente:")
print(df_cerinta2)

# Salvam in fisier
df_cerinta2.to_csv('Cerinta2.csv', index=False)

print("\n>>> Fisierul Cerinta2.csv a fost salvat!")

# ============================================================================
# CERINTA B.1 - Matricea ierarhie (Dendrograma Ward)
# ============================================================================

print("\n\n=== CERINTA B.1 - Analiza de Clusteri (Metoda Ward) ===\n")

# Extragem doar coloanele cu indicatorii pentru clustering
# Eliminam coloanele cu CountryId si Country
df_pentru_clustering = df_calitate[indicatori].copy()

# Eliminam eventualele randuri cu valori lipsa
df_pentru_clustering = df_pentru_clustering.dropna()

print(f"Date pentru clustering: {df_pentru_clustering.shape[0]} tari, {df_pentru_clustering.shape[1]} indicatori")
print()

# Aplicam metoda Ward pentru clustering ierarhic
# linkage calculeaza matricea de legatura (jonctiuni)
# method='ward' = folosim metoda Ward (minimizeaza varianta intra-cluster)
linkage_matrix = linkage(df_pentru_clustering, method='ward')

# Afisam informatii despre jonctiuni
print("Matricea ierarhie (primele 10 jonctiuni):")
print("Format: [Cluster1, Cluster2, Distanta, Nr_instante_in_cluster_nou]")
print()

# Parcurgem primele 10 jonctiuni
for i in range(min(10, len(linkage_matrix))):
    cluster1 = int(linkage_matrix[i, 0])
    cluster2 = int(linkage_matrix[i, 1])
    distanta = linkage_matrix[i, 2]
    nr_instante = int(linkage_matrix[i, 3])

    print(f"Jonctiune {i + 1}: Cluster {cluster1} + Cluster {cluster2} -> "
          f"Distanta = {distanta:.2f}, Instante = {nr_instante}")

print("\n>>> Matricea ierarhie calculata si afisata la consola!")

# ============================================================================
# CERINTA B.2 - Partitia optima (determinare clustere)
# ============================================================================

print("\n\n=== CERINTA B.2 - Partitia Optima ===\n")

# Determinam numarul optim de clustere
# O metoda simpla: alegem 4-5 clustere (poti ajusta)
nr_clustere = 5

# Folosim fcluster pentru a atribui fiecare tara unui cluster
# criterion='maxclust' = specificam numarul de clustere dorit
clustere = fcluster(linkage_matrix, nr_clustere, criterion='maxclust')

print(f"Numar de clustere: {nr_clustere}")
print(f"Distributie tari pe clustere:")

# Afisam cate tari sunt in fiecare cluster
for cluster_id in range(1, nr_clustere + 1):
    nr_tari = np.sum(clustere == cluster_id)
    print(f"  Cluster {cluster_id}: {nr_tari} tari")

# Cream DataFrame cu partitia
# Adaugam si numele tarilor (din indexul original)
df_partitie = pd.DataFrame({
    'Country': df_calitate.loc[df_pentru_clustering.index, 'Country'].values,
    'Cluster': clustere
})

# Salvam in fisier
df_partitie.to_csv('popt.csv', index=False)

print("\nPrimele rezultate din partitie:")
print(df_partitie.head(10))

print("\n>>> Fisierul popt.csv a fost salvat!")

# ============================================================================
# CERINTA B.3 - Histograma pentru o variabila
# ============================================================================

print("\n\n=== CERINTA B.3 - Histograma ===\n")

# Alegem prima variabila pentru histograma (poti schimba)
variabila_aleasa = indicatori[0]  # Air_quality_Carbon_Monoxide

print(f"Variabila selectata pentru histograma: {variabila_aleasa}\n")

# Cream figura
plt.figure(figsize=(10, 6))

# Plotam histograma
plt.hist(df_calitate[variabila_aleasa].dropna(),
         bins=15,  # numar de bare
         color='skyblue',
         edgecolor='black',
         alpha=0.7)

# Adaugam titlu si etichete
plt.title(f'Histograma pentru {variabila_aleasa}', fontsize=14, fontweight='bold')
plt.xlabel('Valoare', fontsize=12)
plt.ylabel('Frecventa', fontsize=12)
plt.grid(axis='y', alpha=0.3)

# Salvam graficul
plt.savefig('histograma.png', dpi=300, bbox_inches='tight')

print(">>> Graficul histograma.png a fost salvat!")

# Afisam graficul
plt.show()

# ============================================================================
# CERINTA C - Calculare KMO global din matricele de corelatie
# ============================================================================

print("\n\n=== CERINTA C - Indexul KMO Global ===\n")

# Citim matricea de corelatie
df_cor = pd.read_csv('g21_1.csv', index_col=0)

print("Matricea de corelatie (g21_1.csv):")
print(df_cor)
print()

# Citim matricea de corelatie partiala
df_cor_partiala = pd.read_csv('g21_2.csv', index_col=0)

print("Matricea de corelatie partiala (g21_2.csv):")
print(df_cor_partiala)
print()

# Calculam KMO global folosind formula:
# KMO = suma(r^2) / (suma(r^2) + suma(p^2))
# unde:
# r = coeficienti de corelatie (fara diagonala)
# p = coeficienti de corelatie partiala (fara diagonala)

# Extragem partea superioara a matricelor (fara diagonala)
# np.triu_indices_from(df_cor, k=1) = indicii triunghiului superior, k=1 exclude diagonala

# Indicii pentru partea superioara (fara diagonala)
indices = np.triu_indices_from(df_cor, k=1)

# Extragem corelatiile (partea superioara)
r = df_cor.values[indices]

# Extragem corelatiile partiale (partea superioara)
p = df_cor_partiala.values[indices]

print(f"Numar de coeficienti de corelatie (fara diagonala): {len(r)}")
print(f"Coeficienti de corelatie: {r}")
print(f"Coeficienti de corelatie partiala: {p}")
print()

# Calculam sumele patratelor
suma_r2 = np.sum(r ** 2)
suma_p2 = np.sum(p ** 2)

print(f"Suma patratelor corelatiilor: {suma_r2:.6f}")
print(f"Suma patratelor corelatiilor partiale: {suma_p2:.6f}")
print()

# Calculam KMO global
kmo_global = suma_r2 / (suma_r2 + suma_p2)

print("=" * 60)
print(f"INDEXUL KMO GLOBAL: {kmo_global:.6f}")
print("=" * 60)
print()

# Interpretare KMO
if kmo_global >= 0.9:
    interpretare = "Excelent"
elif kmo_global >= 0.8:
    interpretare = "Foarte bun"
elif kmo_global >= 0.7:
    interpretare = "Bun"
elif kmo_global >= 0.6:
    interpretare = "Acceptabil"
elif kmo_global >= 0.5:
    interpretare = "Slab"
else:
    interpretare = "Inacceptabil"

print(f"Interpretare: {interpretare}")
print(f"Datele sunt {'potrivite' if kmo_global >= 0.6 else 'nepotrivite'} pentru analiza factoriala.")
print()

print("\n>>> KMO calculat si afisat la consola!")

# ============================================================================
# BONUS: Dendrograma (grafic optional)
# ============================================================================

print("\n\n=== BONUS - Dendrograma (vizualizare clustering) ===\n")

plt.figure(figsize=(15, 8))

# Plotam dendrograma
dendrogram(linkage_matrix,
           labels=df_calitate.loc[df_pentru_clustering.index, 'Country'].values,
           leaf_rotation=90,
           leaf_font_size=8)

plt.title('Dendrograma - Analiza Clustering Ward', fontsize=14, fontweight='bold')
plt.xlabel('Tari', fontsize=12)
plt.ylabel('Distanta', fontsize=12)
plt.tight_layout()

# Salvam dendrograma
plt.savefig('dendrograma.png', dpi=300, bbox_inches='tight')

print(">>> Dendrograma salvata in dendrograma.png!")

plt.show()

print("\n" + "=" * 70)
print("REZOLVARE COMPLETA!")
print("=" * 70)
print("\nFisiere generate:")
print("  1. Cerinta1.csv - Tari cu valori maxime pe indicator")
print("  2. Cerinta2.csv - Tari cu valori maxime pe continent si indicator")
print("  3. popt.csv - Partitia optima (clustere)")
print("  4. histograma.png - Grafic histograma")
print("  5. dendrograma.png - Grafic dendrograma (BONUS)")
print("\nRezultate afisate la consola:")
print("  - Matricea ierarhie (B.1)")
print("  - Indexul KMO global (C)")
print("\nSUCCES LA EXAMEN!")
print("=" * 70)