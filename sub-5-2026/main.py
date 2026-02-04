import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

print("=" * 70)
print("REZOLVARE SUBIECT NETFLIX - ANALIZA DATELOR")
print("=" * 70)

# ============================================================================
# CITIRE DATE
# ============================================================================

print("\n=== PASUL 0: CITIRE FISIERE ===\n")

# Citim fisierul cu datele Netflix
df_netflix = pd.read_csv('Netflix.csv')

# Citim fisierul cu codurile tarilor
df_coduri = pd.read_csv('CoduriTari.csv')

print("Primele randuri din Netflix.csv:")
print(df_netflix.head())
print(f"\nNumar tari: {len(df_netflix)}")
print()

print("Primele randuri din CoduriTari.csv:")
print(df_coduri.head())
print()

# ============================================================================
# CERINTA A.1 - STANDARDIZARE DATE
# ============================================================================

print("\n=== CERINTA A.1: STANDARDIZARE ===\n")

# Identificam coloanele cu indicatorii numerici (de la Librarie la IndiceEducatie)
indicatori = ['Librarie', 'CostLunarBasic', 'CostLunarStandard',
              'CostLunarPremium', 'Internet', 'HDI', 'Venit',
              'IndiceFericire', 'IndiceEducatie']

print(f"Indicatori de standardizat: {indicatori}\n")

# Extragem doar coloanele cu indicatori
X = df_netflix[indicatori].copy()

print("Date originale (primele 3 randuri):")
print(X.head(3))
print()

# STANDARDIZARE
# Formula: z = (x - medie) / abatere_standard
# Folosim StandardScaler din sklearn care face exact asta
scaler = StandardScaler()
X_standardizat = scaler.fit_transform(X)

print("Date dupa standardizare (primele 3 randuri):")
print(X_standardizat[:3])
print()

# Cream un DataFrame cu datele standardizate
df_standardizat = pd.DataFrame(X_standardizat, columns=indicatori)

# Adaugam inapoi coloanele Cod si Tara
df_standardizat['Cod'] = df_netflix['Cod'].values
df_standardizat['Tara'] = df_netflix['Tara'].values

# Reordonam coloanele: Cod, Tara, apoi indicatorii
coloane_ordonate = ['Cod', 'Tara'] + indicatori
df_standardizat = df_standardizat[coloane_ordonate]

# SORTARE DESCRESCATOARE dupa Internet
df_standardizat = df_standardizat.sort_values('Internet', ascending=False)

# Resetam indexul
df_standardizat.reset_index(drop=True, inplace=True)

print("Rezultat final (primele 5 randuri):")
print(df_standardizat.head())
print()

# SALVARE
df_standardizat.to_csv('Cerinta1.csv', index=False)

print(">>> Fisierul Cerinta1.csv a fost salvat!\n")

# ============================================================================
# CERINTA A.2 - COEFICIENTI DE VARIATIE PE CONTINENT
# ============================================================================

print("\n=== CERINTA A.2: COEFICIENTI DE VARIATIE ===\n")

# Unim datele Netflix cu informatiile despre continente
# Folosim coloana 'Cod' ca si cheie
df_complet = df_netflix.merge(df_coduri[['Cod', 'Continent']],
                              on='Cod',
                              how='left')

print("Date dupa unire (primele 3 randuri):")
print(df_complet[['Tara', 'Cod', 'Continent', 'Librarie']].head(3))
print()

# Continentele gasite
continente = df_complet['Continent'].unique()
print(f"Continente gasite: {continente}\n")

# Cream o lista pentru a stoca rezultatele
rezultate_cv = []

# Pentru fiecare continent
for continent in continente:
    # Filtram datele pentru continentul curent
    df_continent = df_complet[df_complet['Continent'] == continent]

    # Cream un dictionar pentru acest continent
    row = {'Continent': continent}

    # Pentru fiecare indicator, calculam Coeficientul de Variatie
    for indicator in indicatori:
        # Extragem valorile pentru acest indicator
        valori = df_continent[indicator]

        # Calculam media
        medie = valori.mean()

        # Calculam abaterea standard (cu ddof=1 pentru esantion)
        std = valori.std(ddof=1)

        # Calculam CV = std / medie
        # Atentie: daca media e 0, punem NaN
        if medie == 0:
            cv = np.nan
        else:
            cv = std / medie

        # Salvam in dictionar
        row[indicator] = cv

    # Adaugam in lista
    rezultate_cv.append(row)

# Cream DataFrame din rezultate
df_cerinta2 = pd.DataFrame(rezultate_cv)

print("Coeficienti de variatie (inainte de sortare):")
print(df_cerinta2)
print()

# SORTARE DESCRESCATOARE dupa coeficientul de variatie al variabilei Librarie
df_cerinta2 = df_cerinta2.sort_values('Librarie', ascending=False)

# Resetam indexul
df_cerinta2.reset_index(drop=True, inplace=True)

print("Rezultat final (sortat dupa Librarie):")
print(df_cerinta2)
print()

# SALVARE
df_cerinta2.to_csv('Cerinta2.csv', index=False)

print(">>> Fisierul Cerinta2.csv a fost salvat!\n")

# ============================================================================
# CERINTA B.1 - ANALIZA IN COMPONENTE PRINCIPALE (PCA)
# ============================================================================

print("\n=== CERINTA B.1: PCA - VARIANTA COMPONENTELOR ===\n")

# Extragem din nou datele numerice (indicatorii)
X_pca = df_netflix[indicatori].copy()

print(f"Dimensiune date pentru PCA: {X_pca.shape}")
print("(57 tari x 9 indicatori)\n")

# STANDARDIZARE (obligatorie pentru PCA)
scaler_pca = StandardScaler()
X_pca_std = scaler_pca.fit_transform(X_pca)

print("Date standardizate pentru PCA (primele 3 randuri):")
print(X_pca_std[:3])
print()

# APLICARE PCA
# Nu specificam n_components, deci se calculeaza toate componentele (9)
pca = PCA()
pca.fit(X_pca_std)

print("PCA aplicat cu succes!\n")

# CALCULARE INDICATORI PENTRU FIECARE COMPONENTA

# 1. Varianta fiecarei componente (eigenvalues)
variances = pca.explained_variance_

# 2. Procentul de varianta explicata de fiecare componenta
pct_variance = pca.explained_variance_ratio_ * 100  # inmultim cu 100 pentru procente

# 3. Varianta cumulata (suma variantelor pana la componenta curenta)
cumulative_variance = np.cumsum(variances)

# 4. Procentul cumulat (suma procentelor pana la componenta curenta)
cumulative_pct = np.cumsum(pct_variance)

# Cream DataFrame cu rezultatele
df_varianta = pd.DataFrame({
    'Componenta': [f'PC{i + 1}' for i in range(len(variances))],
    'Varianta': variances,
    'Procent_Varianta': pct_variance,
    'Varianta_Cumulata': cumulative_variance,
    'Procent_Cumulat': cumulative_pct
})

print("Varianta componentelor principale:")
print(df_varianta)
print()

# SALVARE
df_varianta.to_csv('Varianta.csv', index=False)

print(">>> Fisierul Varianta.csv a fost salvat!\n")

# ============================================================================
# CERINTA B.2 - PLOT VARIANTA CU CRITERII DE RELEVANTA
# ============================================================================

print("\n=== CERINTA B.2: GRAFIC VARIANTA ===\n")

# Cream figura cu subploturi
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# --- GRAFIC 1: SCREE PLOT (variantele componentelor) ---

# Plotam variantele
ax1.plot(range(1, len(variances) + 1), variances,
         marker='o', linewidth=2, markersize=8, color='blue')

# Adaugam linia pentru Criteriul Kaiser (eigenvalue = 1)
ax1.axhline(y=1, color='red', linestyle='--', linewidth=2,
            label='Criteriul Kaiser (eigenvalue = 1)')

# Titlu si etichete
ax1.set_title('Scree Plot - Varianta Componentelor Principale',
              fontsize=14, fontweight='bold')
ax1.set_xlabel('Componenta Principala', fontsize=12)
ax1.set_ylabel('Varianta (Eigenvalue)', fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=10)

# Setam tick-urile pe axa X
ax1.set_xticks(range(1, len(variances) + 1))

# --- GRAFIC 2: PROCENT VARIANTA CUMULATA ---

# Plotam procentul cumulat
ax2.plot(range(1, len(variances) + 1), cumulative_pct,
         marker='s', linewidth=2, markersize=8, color='green')

# Adaugam linia pentru 80% si 90%
ax2.axhline(y=80, color='orange', linestyle='--', linewidth=2,
            label='80% varianta')
ax2.axhline(y=90, color='red', linestyle='--', linewidth=2,
            label='90% varianta')

# Titlu si etichete
ax2.set_title('Procent Varianta Cumulata',
              fontsize=14, fontweight='bold')
ax2.set_xlabel('Componenta Principala', fontsize=12)
ax2.set_ylabel('Procent Varianta Cumulata (%)', fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=10)

# Setam tick-urile pe axa X
ax2.set_xticks(range(1, len(variances) + 1))

# Salvam graficul
plt.tight_layout()
plt.savefig('plot_varianta.png', dpi=300, bbox_inches='tight')

print(">>> Graficul plot_varianta.png a fost salvat!\n")

# Afisam graficul
plt.show()

# ============================================================================
# CERINTA B.3 - CALCUL COMUNALITATI
# ============================================================================

print("\n=== CERINTA B.3: COMUNALITATI ===\n")

# COMUNALITATILE reprezinta cat de bine este explicata fiecare variabila
# de catre TOATE componentele principale

# Formula: Pentru variabila i, comunalitatea = suma(loadings_ij^2)
# unde loadings = corelatia dintre variabilele originale si componentele principale

# Calculam loadings
# loadings = eigenvectors * sqrt(eigenvalues)
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

print("Matricea loadings (primele 3 randuri x 3 coloane):")
print(loadings[:3, :3])
print()

# Comunalitatea pentru fiecare variabila = suma patratelor loadings-urilor
# pe toate componentele
comunalitati = np.sum(loadings ** 2, axis=1)

print("Comunalitati calculate:")
print(comunalitati)
print()

# Cream DataFrame
df_comunalitati = pd.DataFrame({
    'Variabila': indicatori,
    'Comunalitate': comunalitati
})

print("Rezultat final:")
print(df_comunalitati)
print()

# SALVARE
df_comunalitati.to_csv('comm.csv', index=False)

print(">>> Fisierul comm.csv a fost salvat!\n")

# Interpretare
print("INTERPRETARE:")
print("Comunalitatea apropiata de 1 = variabila este foarte bine explicata")
print("Comunalitatea apropiata de 0 = variabila nu este bine explicata")
print()

# ============================================================================
# CERINTA C - ALGORITM ELBOW PENTRU NUMAR OPTIMAL DE CLUSTERI
# ============================================================================

print("\n=== CERINTA C: ALGORITMUL ELBOW ===\n")

# Citim matricea ierarhie
df_h = pd.read_csv('h.csv')

print("Matricea ierarhie (primele 10 jonctiuni):")
print(df_h.head(10))
print()

print("Explicatie coloane:")
print("- Coloana 0-1: clusterii care se unesc")
print("- Coloana 2: distanta intre clusteri (crestere varianta)")
print("- Coloana 3: numar instante in noul cluster")
print()

# Extragem distantele (coloana 2)
distante = df_h.iloc[:, 2].values

print(f"Numar jonctiuni: {len(distante)}\n")

# METODA ELBOW: cautam saltul cel mai mare intre distante consecutive

# Calculam diferentele intre distante consecutive
diferente = np.diff(distante)

print("Diferente intre distante consecutive (ultimele 10):")
print(diferente[-10:])
print()

# Gasim indexul cu diferenta maxima
# Cautam in ULTIMELE jonctiuni (cele mai importante)
# De obicei ne uitam la ultimele 10-15 jonctiuni

ultimele_jonctiuni = 15
idx_max_diff = len(diferente) - ultimele_jonctiuni + np.argmax(diferente[-ultimele_jonctiuni:])

print(f"Index cu diferenta maxima: {idx_max_diff}")
print(f"Diferenta maxima: {diferente[idx_max_diff]:.4f}\n")

# Numarul optimal de clusteri = numarul de jonctiuni ramase dupa jonctiunea cu salt maxim + 1
# Formula: nr_clusteri = numar_total_jonctiuni - index_jonctiune_salt + 1

numar_total_jonctiuni = len(distante)
nr_clusteri_optimal = numar_total_jonctiuni - idx_max_diff

print("=" * 60)
print("REZULTAT ALGORITM ELBOW:")
print("=" * 60)
print(f"Numarul optimal de clusteri: {nr_clusteri_optimal}")
print("=" * 60)
print()

# Vizualizare Elbow (BONUS)
print("\n=== BONUS: GRAFIC ELBOW ===\n")

plt.figure(figsize=(12, 6))

# Plotam distantele
plt.plot(range(1, len(distante) + 1), distante,
         marker='o', linewidth=2, markersize=6, color='blue')

# Marcam punctul cu saltul maxim
plt.scatter([idx_max_diff + 1], [distante[idx_max_diff]],
            color='red', s=200, zorder=5,
            label=f'Salt maxim (jonctiune {idx_max_diff + 1})')

# Titlu si etichete
plt.title('Metoda Elbow - Determinare Numar Optimal de Clusteri',
          fontsize=14, fontweight='bold')
plt.xlabel('Jonctiune (pas in ierarhie)', fontsize=12)
plt.ylabel('Distanta (crestere varianta)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)

# Salvam graficul
plt.savefig('grafic_elbow.png', dpi=300, bbox_inches='tight')

print(">>> Graficul grafic_elbow.png a fost salvat!\n")

# Afisam graficul
plt.show()

# ============================================================================
# REZUMAT FINAL
# ============================================================================

print("\n" + "=" * 70)
print("REZOLVARE COMPLETA!")
print("=" * 70)
print("\nFISIERE GENERATE:")
print("  1. Cerinta1.csv - Date standardizate (sortate dupa Internet)")
print("  2. Cerinta2.csv - Coeficienti de variatie pe continente")
print("  3. Varianta.csv - Varianta componentelor principale")
print("  4. comm.csv - Comunalitati")
print("  5. plot_varianta.png - Grafic varianta si criterii relevanta")
print("  6. grafic_elbow.png - Grafic Elbow (BONUS)")
print("\nREZULTATE AFISATE LA CONSOLA:")
print("  - Numarul optimal de clusteri (Cerinta C)")
print("\nSUCCES LA EXAMEN!")
print("=" * 70)