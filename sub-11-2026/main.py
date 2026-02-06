import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import CCA
from scipy import stats

print("=" * 70)
print("REZOLVARE SUBIECT 11 - PRODUCTIE ENERGIE SI EMISII")
print("=" * 70)

# ============================================================================
# CITIRE DATE
# ============================================================================

print("\n=== PASUL 0: CITIRE FISIERE ===\n")

# Citim fisierele
df_electricity = pd.read_csv('ElectricityProduction.csv')
df_emissions = pd.read_csv('Emissions.csv')
df_population = pd.read_csv('PopulatieEuropa.csv')

print("Primele randuri din ElectricityProduction.csv:")
print(df_electricity.head())
print()

print("Primele randuri din Emissions.csv:")
print(df_emissions.head())
print()

print("Primele randuri din PopulatieEuropa.csv:")
print(df_population.head())
print()

print("Coloane PopulatieEuropa.csv:")
print(df_population.columns.tolist())
print()

# ============================================================================
# CERINTA A.1 - EMISII TOTALE DE PARTICULE
# ============================================================================

print("\n=== CERINTA A.1: EMISII TOTALE DE PARTICULE ===\n")

emisii_coloane = ['AirEmiss', 'Sulphur', 'Nitrogen', 'Ammonia',
                  'NonMeth', 'Partic', 'GreenGE', 'GreenGIE']

print(f"Coloane cu emisii: {emisii_coloane}\n")

df_emisii_tone = df_emissions.copy()

# Convertim GreenGE si GreenGIE din MII TONE in TONE
df_emisii_tone['GreenGE'] = df_emisii_tone['GreenGE'] * 1000
df_emisii_tone['GreenGIE'] = df_emisii_tone['GreenGIE'] * 1000

print("Conversie GreenGE si GreenGIE din mii tone in tone (x1000)")
print()

# Calculam emisiile totale
df_emisii_tone['Emisii_total_tone'] = df_emisii_tone[emisii_coloane].sum(axis=1)

df_cerinta1 = df_emisii_tone[['CountryCode', 'Country', 'Emisii_total_tone']].copy()

print("Rezultate (primele 10 tari):")
print(df_cerinta1.head(10))
print()

df_cerinta1.to_csv('Cerinta1.csv', index=False)

print(">>> Fisierul Cerinta1.csv a fost salvat!\n")

# ============================================================================
# CERINTA A.2 - EMISII LA 100000 LOCUITORI PE REGIUNE
# ============================================================================

print("\n=== CERINTA A.2: EMISII LA 100000 LOCUITORI PE REGIUNE ===\n")

# Redenumim coloana
df_population_renamed = df_population.rename(columns={'Code': 'CountryCode'})

# Unim emisiile cu regiunile
df_complet = df_emisii_tone.merge(
    df_population_renamed[['CountryCode', 'Region', 'Population']],
    on='CountryCode',
    how='left'
)

print("Date dupa unire (primele 3 randuri):")
print(df_complet[['Country', 'Region', 'Population', 'Emisii_total_tone']].head(3))
print()

regiuni = df_complet['Region'].dropna().unique()
print(f"Regiuni gasite: {regiuni}\n")

rezultate_a2 = []

for regiune in regiuni:
    print(f"Procesare regiune: {regiune}")

    df_regiune = df_complet[df_complet['Region'] == regiune]
    populatie_totala = df_regiune['Population'].sum()

    print(f"  Populatie totala: {populatie_totala:,.0f}")

    row = {'Region': regiune}

    for emisie in emisii_coloane:
        total_emisii = df_regiune[emisie].sum()
        emisii_per_100k = (total_emisii / populatie_totala) * 100000

        if emisie in ['GreenGE', 'GreenGIE']:
            row[f'{emisie}_tone'] = emisii_per_100k
        else:
            row[emisie] = emisii_per_100k

    print(f"  Emisii calculate pentru {len(emisii_coloane)} tipuri")
    print()

    rezultate_a2.append(row)

df_cerinta2 = pd.DataFrame(rezultate_a2)

print("Rezultate finale:")
print(df_cerinta2)
print()

df_cerinta2.to_csv('Cerinta2.csv', index=False)

print(">>> Fisierul Cerinta2.csv a fost salvat!\n")

# ============================================================================
# CERINTA B - ANALIZA CANONICA (CU ELIMINARE VALORI LIPSA)
# ============================================================================

print("\n=== CERINTA B: ANALIZA CANONICA ===\n")

# SET 1: productie energie (de la coal la other)
set1_coloane = [col for col in df_electricity.columns
                if col not in ['CountryCode', 'Country']]

# SET 2: emisii
set2_coloane = ['AirEmiss', 'Sulphur', 'Nitrogen', 'Ammonia',
                'NonMeth', 'Partic', 'GreenGE', 'GreenGIE']

print(f"SET 1 (productie energie): {set1_coloane}")
print(f"Numar variabile SET 1: {len(set1_coloane)}\n")

print(f"SET 2 (emisii): {set2_coloane}")
print(f"Numar variabile SET 2: {len(set2_coloane)}\n")

# Unim cele doua seturi
df_analiza = df_electricity.merge(df_emissions, on='CountryCode', how='inner')

print(f"Numar tari dupa unire: {len(df_analiza)}\n")

# VERIFICAM VALORI LIPSA
print("Verificare valori lipsa:")
print(f"  SET 1: {df_analiza[set1_coloane].isna().sum().sum()} valori lipsa")
print(f"  SET 2: {df_analiza[set2_coloane].isna().sum().sum()} valori lipsa")
print()

# ELIMINAM RANDURILE CU VALORI LIPSA (FOARTE IMPORTANT!)
df_analiza_clean = df_analiza.dropna(subset=set1_coloane + set2_coloane)

print(f"Numar tari dupa eliminare valori lipsa: {len(df_analiza_clean)}\n")

# Extragem datele
X = df_analiza_clean[set1_coloane].values
Y = df_analiza_clean[set2_coloane].values

print(f"Dimensiune SET 1 (X): {X.shape}")
print(f"Dimensiune SET 2 (Y): {Y.shape}\n")

# Verificare finala
print("Verificare finala valori NaN:")
print(f"  X contine NaN: {np.isnan(X).any()}")
print(f"  Y contine NaN: {np.isnan(Y).any()}")
print()

# Standardizare
print("Standardizare date...\n")
scaler_x = StandardScaler()
scaler_y = StandardScaler()

X_std = scaler_x.fit_transform(X)
Y_std = scaler_y.fit_transform(Y)

# Numarul de componente canonice
n_components = min(X.shape[1], Y.shape[1])

print(f"Numar componente canonice: {n_components}\n")

# Aplicam CCA
cca = CCA(n_components=n_components)
cca.fit(X_std, Y_std)

print("Analiza canonica aplicata cu succes!\n")

# ============================================================================
# CERINTA B.1 - SCORURI CANONICE
# ============================================================================

print("\n=== CERINTA B.1: SCORURI CANONICE ===\n")

X_c, Y_c = cca.transform(X_std, Y_std)

print(f"Dimensiune scoruri SET 1 (z): {X_c.shape}")
print(f"Dimensiune scoruri SET 2 (u): {Y_c.shape}\n")

# DataFrame pentru scorurile SET 1
df_z = pd.DataFrame(
    X_c,
    columns=[f'z{i + 1}' for i in range(X_c.shape[1])]
)
df_z.insert(0, 'CountryCode', df_analiza_clean['CountryCode'].values)

print("Scoruri canonice SET 1 (primele 5 randuri):")
print(df_z.head())
print()

df_z.to_csv('z.csv', index=False)
print(">>> Fisierul z.csv a fost salvat!\n")

# DataFrame pentru scorurile SET 2
df_u = pd.DataFrame(
    Y_c,
    columns=[f'u{i + 1}' for i in range(Y_c.shape[1])]
)
df_u.insert(0, 'CountryCode', df_analiza_clean['CountryCode'].values)

print("Scoruri canonice SET 2 (primele 5 randuri):")
print(df_u.head())
print()

df_u.to_csv('u.csv', index=False)
print(">>> Fisierul u.csv a fost salvat!\n")

# ============================================================================
# CERINTA B.2 - CORELATII CANONICE
# ============================================================================

print("\n=== CERINTA B.2: CORELATII CANONICE ===\n")

corelatii_canonice = []

for i in range(n_components):
    corr = np.corrcoef(X_c[:, i], Y_c[:, i])[0, 1]
    corelatii_canonice.append(corr)

corelatii_canonice = np.array(corelatii_canonice)

print("Corelatii canonice:")
for i, corr in enumerate(corelatii_canonice):
    print(f"  Pereche {i + 1}: r = {corr:.6f}")
print()

df_r = pd.DataFrame({
    'Pereche_Canonica': [f'Pereche_{i + 1}' for i in range(len(corelatii_canonice))],
    'Corelatie_Canonica': corelatii_canonice
})

print("Rezultat final:")
print(df_r)
print()

df_r.to_csv('r.csv', index=False)
print(">>> Fisierul r.csv a fost salvat!\n")

# ============================================================================
# CERINTA B.3 - TEST BARTLETT
# ============================================================================

print("\n=== CERINTA B.3: TEST BARTLETT ===\n")

n = X.shape[0]
p = X.shape[1]
q = Y.shape[1]

print(f"Numar observatii (n): {n}")
print(f"Numar variabile SET 1 (p): {p}")
print(f"Numar variabile SET 2 (q): {q}")
print()

print("TEST BARTLETT - Semnificatie statistice perechilor canonice:")
print("=" * 60)

p_values = []
radacini_semnificative = 0
prag_semnificatie = 0.01

for k in range(len(corelatii_canonice)):
    # Lambda Wilks
    lambda_wilks = np.prod([1 - corelatii_canonice[i] ** 2
                            for i in range(k, len(corelatii_canonice))])

    # Statistica chi-patrat
    chi_square = -(n - 1 - (p + q + 1) / 2) * np.log(lambda_wilks)

    # Grade de libertate
    df_bartlett = (p - k) * (q - k)

    # P-value
    p_value = 1 - stats.chi2.cdf(chi_square, df_bartlett)
    p_values.append(p_value)

    semnificativ = "DA" if p_value < prag_semnificatie else "NU"

    print(f"Pereche {k + 1}:")
    print(f"  Corelatie canonica: {corelatii_canonice[k]:.6f}")
    print(f"  Chi-patrat: {chi_square:.4f}")
    print(f"  Grade libertate: {df_bartlett}")
    print(f"  P-value: {p_value:.6f}")
    print(f"  Semnificativ (p < {prag_semnificatie}): {semnificativ}")
    print()

    if p_value < prag_semnificatie:
        radacini_semnificative += 1

print("=" * 60)
print("REZULTATE FINALE TEST BARTLETT:")
print("=" * 60)
print(f"Prag de semnificatie: {prag_semnificatie}")
print(f"Numar radacini semnificative: {radacini_semnificative}")
print()

print("P-values pentru fiecare pereche:")
for i, pval in enumerate(p_values):
    print(f"  Pereche {i + 1}: p-value = {pval:.6f}")
print()

# ============================================================================
# CERINTA C - COSINUS CU A TREIA COMPONENTA
# ============================================================================

print("\n=== CERINTA C: COSINUS CU A TREIA COMPONENTA ===\n")

A = np.array([
    [0.4, -0.25, 0.8],
    [0.5, 0.75, 0.6],
    [0.1, 0.75, 0.5]
])

print("Matricea vectorilor proprii (A):")
print(A)
print()

pc3 = A[:, 2]

print("A treia componenta principala (PC3):")
print(pc3)
print()

x = np.array([2, 4, 2])

print("Instanta x:")
print(x)
print()

# CALCULARE COSINUS
produs_scalar = np.dot(x, pc3)
print(f"Produs scalar (x · PC3): {produs_scalar:.4f}")

norma_x = np.linalg.norm(x)
print(f"Norma lui x: {norma_x:.4f}")

norma_pc3 = np.linalg.norm(pc3)
print(f"Norma lui PC3: {norma_pc3:.4f}")
print()

cosinus = produs_scalar / (norma_x * norma_pc3)

print("=" * 60)
print("REZULTAT FINAL:")
print("=" * 60)
print(f"Cosinusul instantei x cu axa celei de-a treia componente: {cosinus:.6f}")
print("=" * 60)
print()

if -1 <= cosinus <= 1:
    print("✓ Rezultatul este valid")
else:
    print("✗ ATENTIE: Rezultatul nu este valid!")
print()

unghi_grade = np.arccos(cosinus) * 180 / np.pi
print(f"Unghi intre x si PC3: {unghi_grade:.2f} grade")
print()

