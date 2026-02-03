import pandas as pd
import numpy as np
from sklearn.decomposition import FactorAnalysis
from factor_analyzer import calculate_kmo
import matplotlib.pyplot as plt

# ============================================================================
# CERINTA A.1 - Calculare diversitate medie si salvare in Cerinta1.csv
# ============================================================================

print("=== CERINTA A.1 ===")

# Citim fisierul cu diversitatile
df_diversitate = pd.read_csv('Diversitate.csv')

# Citim fisierul cu codurile localitatilor
df_coduri = pd.read_csv('Coduri_Localitati.csv')

# Afisam primele randuri ca sa vedem structura datelor
print("Primele randuri din Diversitate.csv:")
print(df_diversitate.head())
print("\nPrimele randuri din Coduri_Localitati.csv:")
print(df_coduri.head())

# Identificam coloanele cu anii (2008-2021)
# Presupunem ca coloanele sunt: Siruta, Localitate, 2008, 2009, ..., 2021
coloane_ani = [str(an) for an in range(2008, 2022)]  # ['2008', '2009', ..., '2021']

# Calculam diversitatea medie pe 2008-2021 (14 ani)
# axis=1 inseamna ca calculam media pe randuri (pe coloane de ani)
df_diversitate['Div_Medie'] = df_diversitate[coloane_ani].mean(axis=1)

# Filtram doar localitatile unde diversitatea medie > diversitatea din 2021
df_filtrat = df_diversitate[df_diversitate['Div_Medie'] > df_diversitate['2021']].copy()

# Selectam doar coloanele cerute: Siruta, Localitate, Div Medie, 2021
df_rezultat_a1 = df_filtrat[['Siruta', 'Localitate', 'Div_Medie', '2021']]

# Salvam in fisier CSV
df_rezultat_a1.to_csv('Cerinta1.csv', index=False)

print(f"\nNumar localitati gasite: {len(df_rezultat_a1)}")
print("\nPrimele rezultate:")
print(df_rezultat_a1.head())
print("\n>>> Fisierul Cerinta1.csv a fost salvat!")

# ============================================================================
# CERINTA A.2 - Anul cu diversitatea medie maxima pe judete
# ============================================================================

print("\n\n=== CERINTA A.2 ===")

# Unim cele doua dataframe-uri dupa codul Siruta
# Astfel, fiecare localitate va avea si judetul asociat
df_complet = pd.merge(df_diversitate, df_coduri[['Siruta', 'Judet']], on='Siruta', how='left')

print("Date dupa unire:")
print(df_complet.head())

# Cream o lista pentru a stoca rezultatele
rezultate_judete = []

# Grupam datele dupa judet
for judet, grup in df_complet.groupby('Judet'):
    # Pentru fiecare an, calculam media diversitatii pe toate localitatile din judet
    medii_pe_ani = {}

    for an in coloane_ani:
        # Calculam media pentru anul curent, ignorand valorile lipsa (NaN)
        medie_an = grup[an].mean()
        medii_pe_ani[an] = medie_an

    # Gasim anul cu media maxima
    an_maxim = max(medii_pe_ani, key=medii_pe_ani.get)

    # Adaugam rezultatul in lista
    rezultate_judete.append({
        'Judet': judet,
        'Anul': an_maxim
    })

# Cream un DataFrame din rezultate
df_rezultat_a2 = pd.DataFrame(rezultate_judete)

# Salvam in fisier
df_rezultat_a2.to_csv('Cerinta2.csv', index=False)

print("\nRezultate pentru fiecare judet:")
print(df_rezultat_a2)
print("\n>>> Fisierul Cerinta2.csv a fost salvat!")

# ============================================================================
# CERINTA B.1 - Calculare indice KMO
# ============================================================================

print("\n\n=== CERINTA B.1 - Indice KMO ===")

# Pregatim datele pentru analiza factoriala
# Extragem doar coloanele cu anii si eliminam valorile lipsa
df_pentru_analiza = df_diversitate[coloane_ani].dropna()

print(f"Dimensiune date pentru analiza: {df_pentru_analiza.shape}")

# Calculam indicele KMO (Kaiser-Meyer-Olkin)
# KMO masoara cat de potrivite sunt datele pentru analiza factoriala
# Valori > 0.6 = acceptabil, > 0.8 = foarte bun
kmo_all, kmo_model = calculate_kmo(df_pentru_analiza)

# kmo_model = KMO general (un singur numar)
# kmo_all = KMO pentru fiecare variabila (pentru fiecare an)

# Cream un DataFrame cu rezultatele KMO
df_kmo = pd.DataFrame({
    'Variabila': coloane_ani + ['KMO_General'],
    'KMO': list(kmo_all) + [kmo_model]
})

# Salvam in fisier
df_kmo.to_csv('KMO.csv', index=False)

print("\nIndicele KMO general:", kmo_model)
print("\nIndicii KMO pe variabile:")
print(df_kmo)
print("\n>>> Fisierul KMO.csv a fost salvat!")

# ============================================================================
# CERINTA B.2 - Analiza factoriala cu rotatie Varimax
# ============================================================================

print("\n\n=== CERINTA B.2 - Analiza Factoriala ===")

# Cream modelul de analiza factoriala
# n_components = numarul de factori (alegem 3-4 factori)
# rotation='varimax' = rotatie Varimax pentru interpretare mai usoara
fa = FactorAnalysis(n_components=4, rotation='varimax', random_state=42)

# Aplicam analiza factoriala pe date
fa.fit(df_pentru_analiza)

# Extragem scorurile factoriale pentru fiecare localitate
# transform() calculeaza valorile factorilor pentru fiecare observatie
scoruri_factoriale = fa.transform(df_pentru_analiza)

# Cream un DataFrame cu scorurile
df_scoruri = pd.DataFrame(
    scoruri_factoriale,
    columns=[f'Factor{i + 1}' for i in range(scoruri_factoriale.shape[1])]
)

# Salvam scorurile in fisier
df_scoruri.to_csv('f.csv', index=False)

print(f"Dimensiune scoruri factoriale: {df_scoruri.shape}")
print("\nPrimele scoruri factoriale:")
print(df_scoruri.head())
print("\n>>> Fisierul f.csv a fost salvat!")

# ============================================================================
# CERINTA B.3 - Plot scoruri pentru primii 2 factori
# ============================================================================

print("\n\n=== CERINTA B.3 - Grafic Scoruri ===")

# Cream figura pentru grafic
plt.figure(figsize=(10, 8))

# Plotam scorurile pentru primii doi factori
# Factor1 pe axa X, Factor2 pe axa Y
plt.scatter(df_scoruri['Factor1'], df_scoruri['Factor2'], alpha=0.5, s=20)

# Adaugam titlu si etichete
plt.title('Plotul scorurilor pentru primii doi factori comuni', fontsize=14)
plt.xlabel('Factor 1', fontsize=12)
plt.ylabel('Factor 2', fontsize=12)

# Adaugam grila pentru vizualizare mai buna
plt.grid(True, alpha=0.3)

# Adaugam linii pentru axe
plt.axhline(y=0, color='k', linewidth=0.5)
plt.axvline(x=0, color='k', linewidth=0.5)

# Salvam graficul
plt.savefig('plot_factori.png', dpi=300, bbox_inches='tight')
print(">>> Graficul a fost salvat in plot_factori.png!")

# Afisam graficul
plt.show()

# ============================================================================
# CERINTA C - Calculare cosinus intre instanta si prima componenta principala
# ============================================================================

print("\n\n=== CERINTA C - Cosinus cu prima componenta principala ===")

# Citim matricea vectorilor proprii din fisierul a.csv
df_pca = pd.read_csv('a.csv')

print("Matricea vectorilor proprii:")
print(df_pca)

# Extragem prima coloana (PC1 = prima componenta principala = axa 1)
pc1 = df_pca['PC1'].values

print("\nPrima componenta principala (PC1):")
print(pc1)

# Definim instanta x
x = np.array([1, 2, -3, 3, 0])

print("\nInstanta x:")
print(x)

# Calculam produsul scalar intre x si PC1
# Produsul scalar = x1*pc1_1 + x2*pc1_2 + ... + x5*pc1_5
produs_scalar = np.dot(x, pc1)

print(f"\nProdus scalar (x · PC1): {produs_scalar}")

# Calculam norma (lungimea) vectorului x
# Norma = sqrt(x1^2 + x2^2 + ... + x5^2)
norma_x = np.linalg.norm(x)

print(f"Norma vectorului x: {norma_x}")

# Calculam norma vectorului PC1
norma_pc1 = np.linalg.norm(pc1)

print(f"Norma vectorului PC1: {norma_pc1}")

# Calculam cosinusul unghiului
# cos(theta) = (x · PC1) / (||x|| * ||PC1||)
cosinus = produs_scalar / (norma_x * norma_pc1)

print(f"\n{'=' * 60}")
print(f"REZULTAT FINAL:")
print(f"Cosinusul unghiului intre x si prima componenta principala: {cosinus}")
print(f"{'=' * 60}")

# Verificare: cosinusul trebuie sa fie intre -1 si 1
if -1 <= cosinus <= 1:
    print("✓ Rezultatul este valid (cosinus in [-1, 1])")
else:
    print("✗ ATENTIE: Rezultatul nu este valid!")

print("\n=== REZOLVARE COMPLETA ===")