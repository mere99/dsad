import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FactorAnalysis
from scipy.cluster.hierarchy import fcluster
import matplotlib.pyplot as plt

print("=" * 70)
print("REZOLVARE SUBIECT 8 - NETFLIX")
print("=" * 70)

# ============================================================================
# CITIRE DATE
# ============================================================================

print("\n=== PASUL 0: CITIRE FISIERE ===\n")

# Citim fisierul Netflix
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

# Identificam variabilele numerice (de la Librarie la IndiceEducatie)
variabile_numerice = ['Librarie', 'CostLunarBasic', 'CostLunarStandard',
                      'CostLunarPremium', 'Internet', 'HDI', 'Venit',
                      'IndiceFericire', 'IndiceEducatie']

print(f"Variabile numerice: {variabile_numerice}\n")

# ============================================================================
# CERINTA A.1 - TARA CU VALOAREA MAXIMA PENTRU FIECARE VARIABILA
# ============================================================================

print("\n=== CERINTA A.1: TARA CU VALOARE MAXIMA PE VARIABILA ===\n")

# Cream o lista pentru rezultate
rezultate_a1 = []

# Pentru fiecare variabila numerica
for variabila in variabile_numerice:
    # Gasim indexul cu valoarea maxima
    idx_max = df_netflix[variabila].idxmax()

    # Extragem codul tarii cu valoarea maxima
    tara_max = df_netflix.loc[idx_max, 'Cod']

    # Adaugam in lista
    rezultate_a1.append({
        'Variabila': variabila,
        'Tara': tara_max
    })

    print(f"{variabila}: {tara_max}")

# Cream DataFrame
df_cerinta1 = pd.DataFrame(rezultate_a1)

print("\nRezultat final:")
print(df_cerinta1)
print()

# SALVARE
df_cerinta1.to_csv('Cerinta1.csv', index=False)

print(">>> Fisierul Cerinta1.csv a fost salvat!\n")

# ============================================================================
# CERINTA A.2 - MATRICI DE CORELATIE PE CONTINENT
# ============================================================================

print("\n=== CERINTA A.2: MATRICI DE CORELATIE PE CONTINENT ===\n")

# Unim datele Netflix cu informatiile despre continente
df_complet = df_netflix.merge(df_coduri[['Cod', 'Continent']],
                              on='Cod',
                              how='left')

print("Date dupa unire (primele 3 randuri):")
print(df_complet[['Tara', 'Cod', 'Continent', 'Librarie']].head(3))
print()

# Continentele gasite
continente = df_complet['Continent'].dropna().unique()
print(f"Continente gasite: {continente}\n")

# Pentru fiecare continent
for continent in continente:
    print(f"Procesare continent: {continent}")

    # Filtram datele pentru continentul curent
    df_continent = df_complet[df_complet['Continent'] == continent]

    # Extragem doar variabilele numerice
    df_numeric = df_continent[variabile_numerice]

    # Calculam matricea de corelatie
    matrice_corelatie = df_numeric.corr()

    print(f"  Dimensiune matrice: {matrice_corelatie.shape}")

    # Salvam matricea in fisier CSV
    # Numele fisierului: nume_continent.csv
    nume_fisier = f"{continent}.csv"
    matrice_corelatie.to_csv(nume_fisier)

    print(f"  >>> Salvat in {nume_fisier}")
    print()

print(">>> Toate matricile de corelatie au fost salvate!\n")

# ============================================================================
# CERINTA B - ANALIZA FACTORIALA (CU STANDARDIZARE!)
# ============================================================================

print("\n=== CERINTA B: ANALIZA FACTORIALA ===\n")

# Extragem datele numerice (eliminam valorile lipsa)
X = df_netflix[variabile_numerice].dropna()

print(f"Dimensiune date pentru analiza factoriala: {X.shape}")
print()

# STANDARDIZARE (FOARTE IMPORTANT!)
print("Standardizare date...\n")
scaler = StandardScaler()
X_standardizat = scaler.fit_transform(X)

# Aplicam analiza factoriala fara rotatie
# Alegem 3 factori (pentru a avea cercul corelatiilor pentru primii 3)
n_factori = 3

print(f"Numar de factori: {n_factori}\n")

# Cream modelul de analiza factoriala
# rotation=None = fara rotatie
fa = FactorAnalysis(n_components=n_factori, rotation=None, random_state=42)

# Aplicam analiza PE DATELE STANDARDIZATE
fa.fit(X_standardizat)

print("Analiza factoriala aplicata cu succes!\n")

# ============================================================================
# CERINTA B.1 - CORELATIILE FACTORIALE (LOADINGS)
# ============================================================================

print("\n=== CERINTA B.1: CORELATIILE FACTORIALE ===\n")

# Extragem loadings (corelatiile factoriale)
# In sklearn, components_ contine deja loadings-urile
loadings = fa.components_.T

print("Loadings (corelatii factoriale) - primele 3 variabile:")
print(loadings[:3])
print()

# Cream DataFrame cu corelatiile
df_corelatii = pd.DataFrame(
    loadings,
    index=variabile_numerice,
    columns=[f'Factor{i + 1}' for i in range(n_factori)]
)

print("Rezultat final:")
print(df_corelatii)
print()

# Verificare: corelatiile trebuie sa fie intre -1 si 1
print("Verificare:")
print(f"  Valoare minima: {loadings.min():.4f}")
print(f"  Valoare maxima: {loadings.max():.4f}")
print()

# SALVARE
df_corelatii.to_csv('R.csv')

print(">>> Fisierul R.csv a fost salvat!\n")

# ============================================================================
# CERINTA B.2 - CERCUL CORELATIILOR (3 GRAFICE)
# ============================================================================

print("\n=== CERINTA B.2: CERCUL CORELATIILOR ===\n")


# Functie pentru a desena cercul
def draw_circle():
    """Deseneaza un cerc unitate"""
    theta = np.linspace(0, 2 * np.pi, 100)
    x_circle = np.cos(theta)
    y_circle = np.sin(theta)
    return x_circle, y_circle


# GRAFIC 1: Factor 1 vs Factor 2
plt.figure(figsize=(10, 10))

# Desenam cercul
x_circle, y_circle = draw_circle()
plt.plot(x_circle, y_circle, 'k--', linewidth=1, alpha=0.3)

# Plotam vectorii (sagetile) pentru fiecare variabila
for i, var in enumerate(variabile_numerice):
    # Coordonatele variantei pe primii 2 factori
    x = loadings[i, 0]
    y = loadings[i, 1]

    # Desenam sageata
    plt.arrow(0, 0, x, y,
              head_width=0.05, head_length=0.05,
              fc='blue', ec='blue', linewidth=2)

    # Adaugam eticheta
    plt.text(x * 1.15, y * 1.15, var,
             fontsize=9, ha='center', va='center')

# Setari grafic
plt.xlim(-1.2, 1.2)
plt.ylim(-1.2, 1.2)
plt.axhline(y=0, color='k', linewidth=0.5)
plt.axvline(x=0, color='k', linewidth=0.5)
plt.grid(True, alpha=0.3)
plt.title('Cercul Corelatiilor - Factor 1 vs Factor 2',
          fontsize=14, fontweight='bold')
plt.xlabel('Factor 1', fontsize=12)
plt.ylabel('Factor 2', fontsize=12)
plt.axis('equal')

plt.savefig('cerc_corelatii_F1_F2.png', dpi=300, bbox_inches='tight')
print(">>> Graficul cerc_corelatii_F1_F2.png a fost salvat!")

plt.show()

# GRAFIC 2: Factor 1 vs Factor 3
plt.figure(figsize=(10, 10))

x_circle, y_circle = draw_circle()
plt.plot(x_circle, y_circle, 'k--', linewidth=1, alpha=0.3)

for i, var in enumerate(variabile_numerice):
    x = loadings[i, 0]
    y = loadings[i, 2]  # Factor 3

    plt.arrow(0, 0, x, y,
              head_width=0.05, head_length=0.05,
              fc='green', ec='green', linewidth=2)

    plt.text(x * 1.15, y * 1.15, var,
             fontsize=9, ha='center', va='center')

plt.xlim(-1.2, 1.2)
plt.ylim(-1.2, 1.2)
plt.axhline(y=0, color='k', linewidth=0.5)
plt.axvline(x=0, color='k', linewidth=0.5)
plt.grid(True, alpha=0.3)
plt.title('Cercul Corelatiilor - Factor 1 vs Factor 3',
          fontsize=14, fontweight='bold')
plt.xlabel('Factor 1', fontsize=12)
plt.ylabel('Factor 3', fontsize=12)
plt.axis('equal')

plt.savefig('cerc_corelatii_F1_F3.png', dpi=300, bbox_inches='tight')
print(">>> Graficul cerc_corelatii_F1_F3.png a fost salvat!")

plt.show()

# GRAFIC 3: Factor 2 vs Factor 3
plt.figure(figsize=(10, 10))

x_circle, y_circle = draw_circle()
plt.plot(x_circle, y_circle, 'k--', linewidth=1, alpha=0.3)

for i, var in enumerate(variabile_numerice):
    x = loadings[i, 1]  # Factor 2
    y = loadings[i, 2]  # Factor 3

    plt.arrow(0, 0, x, y,
              head_width=0.05, head_length=0.05,
              fc='red', ec='red', linewidth=2)

    plt.text(x * 1.15, y * 1.15, var,
             fontsize=9, ha='center', va='center')

plt.xlim(-1.2, 1.2)
plt.ylim(-1.2, 1.2)
plt.axhline(y=0, color='k', linewidth=0.5)
plt.axvline(x=0, color='k', linewidth=0.5)
plt.grid(True, alpha=0.3)
plt.title('Cercul Corelatiilor - Factor 2 vs Factor 3',
          fontsize=14, fontweight='bold')
plt.xlabel('Factor 2', fontsize=12)
plt.ylabel('Factor 3', fontsize=12)
plt.axis('equal')

plt.savefig('cerc_corelatii_F2_F3.png', dpi=300, bbox_inches='tight')
print(">>> Graficul cerc_corelatii_F2_F3.png a fost salvat!")

plt.show()

print()

# ============================================================================
# CERINTA B.3 - COMUNALITATI SI VARIANTE SPECIFICE
# ============================================================================

print("\n=== CERINTA B.3: COMUNALITATI SI VARIANTE SPECIFICE ===\n")

# COMUNALITATEA = suma patratelor loadings-urilor pe toti factorii
# Pentru fiecare variabila: h^2 = sum(loading_i^2)
comunalitati = np.sum(loadings ** 2, axis=1)

# VARIANTA SPECIFICA = 1 - comunalitate
# psi = 1 - h^2
variante_specifice = 1 - comunalitati

print("Comunalitati si variante specifice:")
for i, var in enumerate(variabile_numerice):
    print(f"{var}: comunalitate = {comunalitati[i]:.4f}, "
          f"varianta specifica = {variante_specifice[i]:.4f}")
print()

# Cream DataFrame
df_comm_spec = pd.DataFrame({
    'Variabila': variabile_numerice,
    'Comunalitate': comunalitati,
    'Varianta_Specifica': variante_specifice
})

print("Rezultat final:")
print(df_comm_spec)
print()

# SALVARE
df_comm_spec.to_csv('comm_spec.csv', index=False)

print(">>> Fisierul comm_spec.csv a fost salvat!\n")

# ============================================================================
# CERINTA C - DISTANTE DE SECTIONARE PENTRU 4 CLUSTERI
# ============================================================================

print("\n=== CERINTA C: DISTANTE DE SECTIONARE (4 CLUSTERI) ===\n")

# Citim matricea ierarhie
df_h = pd.read_csv('h.csv')

print("Matricea ierarhie (primele 10 jonctiuni):")
print(df_h.head(10))
print()

print("Matricea ierarhie (ultimele 10 jonctiuni):")
print(df_h.tail(10))
print()

# Extragem distantele (coloana 2, indexare de la 0)
# Structura: [cluster1, cluster2, distanta, nr_instante]
distante = df_h.iloc[:, 2].values

print(f"Numar total jonctiuni: {len(distante)}\n")

# Pentru 4 clusteri, avem nevoie sa "taiem" dendrograma
# astfel incat sa ramana exact 4 clusteri

# Numarul de instante = nr jonctiuni + 1
nr_instante = len(distante) + 1

print(f"Numar instante: {nr_instante}")

# Pentru 4 clusteri:
# - jonctiunea[-4] = reduce de la 5 la 4 clusteri (DIST MIN)
# - jonctiunea[-3] = reduce de la 4 la 3 clusteri (DIST MAX)

# Valoarea MINIMA = distanta de la jonctiunea care creeaza al 4-lea cluster
dist_min = distante[-4]

# Valoarea MAXIMA = distanta de la jonctiunea care reduce de la 4 la 3
dist_max = distante[-3]

print(f"\nPentru a obtine 4 clusteri:")
print(f"  Jonctiunea care reduce de la 5 la 4 clusteri: distanta = {dist_min:.4f}")
print(f"  Jonctiunea care reduce de la 4 la 3 clusteri: distanta = {dist_max:.4f}")
print()

print("=" * 60)
print("REZULTAT FINAL:")
print("=" * 60)
print(f"Valoarea MINIMA a distantei de sectionare: {dist_min:.4f}")
print(f"Valoarea MAXIMA a distantei de sectionare: {dist_max:.4f}")
print("=" * 60)
print()

print("Interpretare:")
print(f"  - Daca taiem la distanta < {dist_min:.4f}: avem 5 sau mai multi clusteri")
print(f"  - Daca taiem la distanta intre {dist_min:.4f} si {dist_max:.4f}: avem EXACT 4 clusteri")
print(f"  - Daca taiem la distanta > {dist_max:.4f}: avem 3 sau mai putini clusteri")
print()

