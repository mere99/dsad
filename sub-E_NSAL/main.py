import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# Setam encoding pentru caracterele romanesti
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("=" * 70)
print("REZOLVARE SUBIECT 4 - ANGAJATI SI CLASIFICARE PACIENTI")
print("=" * 70)

# ============================================================================
# CERINTA A.1 - Anul cu cei mai multi angajati pentru fiecare localitate
# ============================================================================

print("\n=== CERINTA A.1 ===\n")

# Citim fisierul cu numarul de angajati
df_angajati = pd.read_csv('E_NSAL_2008-2021.csv')

print("Primele randuri din E_NSAL_2008-2021.csv:")
print(df_angajati.head())
print()

# Citim fisierul cu informatii despre localitati
df_localitati = pd.read_csv('PopulatieLocalitati.csv')

print("Primele randuri din PopulatieLocalitati.csv:")
print(df_localitati.head())
print()

# Identificam coloanele cu anii (2008-2021)
coloane_ani = [str(an) for an in range(2008, 2022)]  # ['2008', '2009', ..., '2021']

print(f"Ani gasiti: {coloane_ani}\n")

# Cream o lista pentru a stoca rezultatele
rezultate_a1 = []

# Pentru fiecare localitate (fiecare rand)
for idx, rand in df_angajati.iterrows():
    # Extragem valorile pentru toti anii
    valori_ani = rand[coloane_ani]

    # Gasim anul cu cei mai multi angajati
    # idxmax() returneaza numele coloanei (anul) cu valoarea maxima
    an_maxim = valori_ani.idxmax()

    # Extragem codul SIRUTA
    siruta = rand['SIRUTA']

    # Gasim denumirea localitatii din celelalte date
    localitate = df_localitati[df_localitati['SIRUTA'] == siruta]['Localitate'].values

    if len(localitate) > 0:
        localitate = localitate[0]
    else:
        localitate = f"Localitate_{siruta}"  # fallback daca nu gasim

    # Adaugam in lista de rezultate
    rezultate_a1.append({
        'Siruta': siruta,
        'Localitate': localitate,
        'Anul': an_maxim
    })

# Cream DataFrame din rezultate
df_cerinta1 = pd.DataFrame(rezultate_a1)

# Salvam in fisier
df_cerinta1.to_csv('Cerinta1.csv', index=False)

print("Rezultate (primele 10 localitati):")
print(df_cerinta1.head(10))

print("\n>>> Fisierul Cerinta1.csv a fost salvat!")

# ============================================================================
# CERINTA A.2 - Rata ocuparii populatiei pe judete
# ============================================================================

print("\n\n=== CERINTA A.2 ===\n")

# Unim datele de angajati cu informatiile despre localitati (judet, populatie)
df_complet = pd.merge(df_angajati, df_localitati[['SIRUTA', 'Judet', 'Populatie']],
                      on='SIRUTA', how='left')

print("Date dupa unire:")
print(df_complet.head())
print()

# Grupam pe judete
judete = df_complet['Judet'].unique()
print(f"Judete gasite: {judete}\n")

# Cream un dictionar pentru a stoca ratele de ocupare
rezultate_judete = []

# Pentru fiecare judet
for judet in judete:
    # Filtram datele pentru judetul curent
    df_judet = df_complet[df_complet['Judet'] == judet]

    # Calculam populatia totala a judetului
    populatie_totala = df_judet['Populatie'].sum()

    # Cream un dictionar pentru acest judet
    rata_judet = {'Judet': judet}

    # Pentru fiecare an, calculam rata de ocupare
    rate_pe_ani = []

    for an in coloane_ani:
        # Suma angajatilor din toate localitatile judetului pentru anul curent
        angajati_total = df_judet[an].sum()

        # Calculam rata de ocupare = angajati / populatie
        rata = angajati_total / populatie_totala if populatie_totala > 0 else 0

        # Salvam in dictionar
        rata_judet[an] = round(rata, 3)
        rate_pe_ani.append(rata)

    # Calculam rata medie pe toti anii
    rata_medie = np.mean(rate_pe_ani)
    rata_judet['RataMedie'] = round(rata_medie, 3)

    # Adaugam in lista
    rezultate_judete.append(rata_judet)

# Cream DataFrame din rezultate
df_cerinta2 = pd.DataFrame(rezultate_judete)

# Sortam descrescator dupa RataMedie
df_cerinta2 = df_cerinta2.sort_values('RataMedie', ascending=False)

# Resetam indexul
df_cerinta2.reset_index(drop=True, inplace=True)

# Salvam in fisier
df_cerinta2.to_csv('Cerinta2.csv', index=False)

print("Rezultate (sortate descrescator dupa RataMedie):")
print(df_cerinta2)

print("\n>>> Fisierul Cerinta2.csv a fost salvat!")

# ============================================================================
# CERINTA B.1 - Analiza Liniara Discriminanta (LDA)
# ============================================================================

print("\n\n=== CERINTA B.1 - Analiza Liniara Discriminanta ===\n")

# Citim datele pacientilor pentru antrenare
df_pacienti = pd.read_csv('Pacienti.csv')

print("Primele randuri din Pacienti.csv:")
print(df_pacienti.head())
print()

print(f"Numar pacienti: {len(df_pacienti)}")
print(f"Distributie clase:")
print(df_pacienti['DECISION'].value_counts())
print()

# Selectam variabilele predictor (de la L_CORE la BP_ST)
# Excludem Id si DECISION
coloane_predictor = ['L_CORE', 'L_SURF', 'L_O2', 'L_BP',
                     'H_CORE', 'H_SURF', 'H_O2', 'H_BP',
                     'M_CORE', 'M_SURF', 'M_O2', 'M_BP', 'BP_ST']

# Extragem variabilele predictor (X) si tinta (y)
X_train = df_pacienti[coloane_predictor]
y_train = df_pacienti['DECISION']

print(f"Dimensiune date de antrenare: {X_train.shape}")
print()

# Cream modelul LDA
# n_components=2 pentru a avea 2 axe discriminante (pentru grafic 2D)
lda = LinearDiscriminantAnalysis(n_components=2)

# Antrenam modelul
lda.fit(X_train, y_train)

# Calculam scorurile discriminante pentru datele de antrenare
scoruri_discriminante = lda.transform(X_train)

print(f"Dimensiune scoruri discriminante: {scoruri_discriminante.shape}")
print()

# Cream DataFrame cu scorurile
# LD1 = prima axa discriminanta, LD2 = a doua axa discriminanta
df_scoruri = pd.DataFrame(
    scoruri_discriminante,
    columns=['LD1', 'LD2']
)

# Adaugam si clasa (DECISION)
df_scoruri['DECISION'] = y_train.values

# Salvam scorurile in fisier z.csv
df_scoruri_salvare = df_scoruri[['LD1', 'LD2']].copy()
df_scoruri_salvare.to_csv('z.csv', index=False)

print("Primele scoruri discriminante:")
print(df_scoruri.head(10))

print("\n>>> Fisierul z.csv a fost salvat!")

# ============================================================================
# CERINTA B.2 - Graficul scorurilor discriminante
# ============================================================================

print("\n\n=== CERINTA B.2 - Grafic Scoruri Discriminante ===\n")

# Cream figura
plt.figure(figsize=(10, 8))

# Definim culori pentru fiecare clasa
culori = {'I': 'red', 'S': 'blue', 'A': 'green'}
etichete = {'I': 'Ingrijire Intensiva', 'S': 'Externat', 'A': 'Ingrijire Generala'}

# Plotam scorurile pentru fiecare clasa
for clasa in ['I', 'S', 'A']:
    # Filtram datele pentru clasa curenta
    df_clasa = df_scoruri[df_scoruri['DECISION'] == clasa]

    # Plotam punctele
    plt.scatter(df_clasa['LD1'], df_clasa['LD2'],
                c=culori[clasa],
                label=etichete[clasa],
                alpha=0.6,
                s=50,
                edgecolors='black',
                linewidth=0.5)

# Adaugam titlu si etichete
plt.title('Scoruri Discriminante - Primele Doua Axe', fontsize=14, fontweight='bold')
plt.xlabel('Prima Axa Discriminanta (LD1)', fontsize=12)
plt.ylabel('A Doua Axa Discriminanta (LD2)', fontsize=12)

# Adaugam legenda
plt.legend(fontsize=10)

# Adaugam grila
plt.grid(True, alpha=0.3)

# Adaugam linii pentru axe
plt.axhline(y=0, color='k', linewidth=0.5)
plt.axvline(x=0, color='k', linewidth=0.5)

# Salvam graficul
plt.savefig('grafic_scoruri_discriminante.png', dpi=300, bbox_inches='tight')

print(">>> Graficul grafic_scoruri_discriminante.png a fost salvat!")

# Afisam graficul
plt.show()

# ============================================================================
# CERINTA B.3 - Matricea de confuzie si indicatori de acuratete
# ============================================================================

print("\n\n=== CERINTA B.3 - Evaluare Model ===\n")

# Facem predictii pe setul de antrenare
y_pred_train = lda.predict(X_train)

# Calculam matricea de confuzie
matrice_confuzie = confusion_matrix(y_train, y_pred_train, labels=['A', 'I', 'S'])

print("Matricea de Confuzie:")
print("Randuri = Clase reale, Coloane = Clase prezise")
print("Ordine: A (Ingrijire Generala), I (Intensiva), S (Externat)")
print()
print(matrice_confuzie)
print()

# Salvam matricea de confuzie in fisier
df_matrice = pd.DataFrame(
    matrice_confuzie,
    index=['A_real', 'I_real', 'S_real'],
    columns=['A_pred', 'I_pred', 'S_pred']
)
df_matrice.to_csv('matc.csv')

print(">>> Fisierul matc.csv a fost salvat!")
print()

# Calculam indicatori de acuratete
acuratete = accuracy_score(y_train, y_pred_train)

# Pentru precision, recall, f1-score folosim average='weighted' pentru multi-class
precizie = precision_score(y_train, y_pred_train, average='weighted', zero_division=0)
recall = recall_score(y_train, y_pred_train, average='weighted', zero_division=0)
f1 = f1_score(y_train, y_pred_train, average='weighted', zero_division=0)

print("=" * 60)
print("INDICATORI DE ACURATETE (afisati la consola):")
print("=" * 60)
print(f"Acuratete (Accuracy):        {acuratete:.4f} ({acuratete * 100:.2f}%)")
print(f"Precizie (Precision):        {precizie:.4f} ({precizie * 100:.2f}%)")
print(f"Sensibilitate (Recall):      {recall:.4f} ({recall * 100:.2f}%)")
print(f"Scor F1 (F1-Score):          {f1:.4f} ({f1 * 100:.2f}%)")
print("=" * 60)
print()

# Calculam si indicatori per clasa
print("Indicatori per clasa:")
print()

for clasa in ['A', 'I', 'S']:
    # Cream o varianta binara (clasa curenta vs. rest)
    y_bin_true = (y_train == clasa).astype(int)
    y_bin_pred = (y_pred_train == clasa).astype(int)

    prec = precision_score(y_bin_true, y_bin_pred, zero_division=0)
    rec = recall_score(y_bin_true, y_bin_pred, zero_division=0)
    f1_clasa = f1_score(y_bin_true, y_bin_pred, zero_division=0)

    print(f"Clasa {clasa} ({etichete[clasa]}):")
    print(f"  Precizie: {prec:.4f}")
    print(f"  Recall:   {rec:.4f}")
    print(f"  F1-Score: {f1_clasa:.4f}")
    print()

# ============================================================================
# BONUS - Aplicare model pe date noi (Pacienti_apply.csv)
# ============================================================================

print("\n=== BONUS - Aplicare Model pe Date Noi ===\n")

# Citim datele noi
df_apply = pd.read_csv('Pacienti_apply.csv')

print("Pacienti noi (fara diagnostic):")
print(df_apply.head())
print()

# Extragem variabilele predictor
X_apply = df_apply[coloane_predictor]

# Facem predictii
predictii = lda.predict(X_apply)

# Adaugam predictiile la DataFrame
df_apply['DECISION_PREDICTED'] = predictii

print("Predictii pentru pacienti noi:")
print(df_apply[['Id', 'DECISION_PREDICTED']])
print()

# Salvam predictiile
df_apply[['Id', 'DECISION_PREDICTED']].to_csv('predictii_pacienti.csv', index=False)

print(">>> Fisierul predictii_pacienti.csv a fost salvat!")

# ============================================================================
# CERINTA C - Numar de componente principale semnificative (Criteriul Kaiser)
# ============================================================================

print("\n\n=== CERINTA C - Criteriul Kaiser ===\n")

# Citim fisierul cu comunalitatile
df_comm = pd.read_csv('comm.csv')

print("Comunitatile:")
print(df_comm)
print()

# Extragem comunalitatile
comunitati = df_comm['Communality'].values

print(f"Comunitati: {comunitati}")
print()

# Criteriul Kaiser: componenta principala este semnificativa daca eigenvalue > 1
# Comunalitatea = suma patratelor incarcarilor pe toate componentele
# Pentru a estima eigenvalues din comunitati, putem folosi aproximarea:
# Daca avem k variabile si comunitatile, atunci:
# Eigenvalue_i ≈ suma comunalitatilor pentru componenta i

# O abordare mai simpla:
# Numarul de componente semnificative ≈ numarul de comunitati > 1/k
# unde k = numarul de variabile

k = len(comunitati)
print(f"Numar variabile (k): {k}")
print()

# Alta abordare: folosim criteriul Kaiser direct
# Eigenvalue > 1 inseamna ca componenta explica mai mult decat o variabila originala

# Deoarece avem doar comunalitatile si nu eigenvalues-urile,
# vom estima: daca comunalitatea medie este mare (>0.8), atunci probabil avem 1-2 componente
# Aceasta este o aproximare

medie_comunitati = np.mean(comunitati)
print(f"Comunalitate medie: {medie_comunitati:.4f}")
print()

# Estimare simpla: numar de componente ≈ numar de comunitati > 0.7
comp_semnificative_estimare = np.sum(comunitati > 0.7)

print("=" * 60)
print("CRITERIUL KAISER - ESTIMARE:")
print("=" * 60)
print(f"Comunalitati > 0.7: {comp_semnificative_estimare}")
print()
print("Notă: Pentru o determinare exacta a numarului de componente")
print("semnificative conform criteriului Kaiser (eigenvalue > 1),")
print("ar fi nevoie de matricea de corelatie sau eigenvalues directe.")
print()
print("Estimare bazata pe comunitati mari (>0.7):")
print(f"Numar de componente principale semnificative: {comp_semnificative_estimare}")
print()

# O abordare mai riguroasa daca presupunem o distributie uniforma
# Suma comunalitatilor = suma eigenvalues pentru componentele retinute
suma_comm = np.sum(comunitati)
print(f"Suma comunalitatilor: {suma_comm:.4f}")

# Daca fiecare componenta ar avea eigenvalue = suma_comm / k_componente
# Vrem eigenvalue > 1, deci k_componente < suma_comm
nr_comp_estimat = int(np.floor(suma_comm))

print(f"Estimare alternativa (suma comunitati): {nr_comp_estimat}")
print("=" * 60)

# ============================================================================
# BONUS - Vizualizare Matricea de Confuzie (Heatmap)
# ============================================================================

print("\n\n=== BONUS - Vizualizare Matricea de Confuzie ===\n")

plt.figure(figsize=(8, 6))

# Cream heatmap
sns.heatmap(matrice_confuzie,
            annot=True,  # afiseaza valorile in celule
            fmt='d',  # format intreg
            cmap='Blues',  # culori
            xticklabels=['A (Generala)', 'I (Intensiva)', 'S (Externat)'],
            yticklabels=['A (Generala)', 'I (Intensiva)', 'S (Externat)'])

plt.title('Matricea de Confuzie - Clasificare Pacienti', fontsize=14, fontweight='bold')
plt.ylabel('Clasa Reala', fontsize=12)
plt.xlabel('Clasa Prezisa', fontsize=12)

# Salvam graficul
plt.savefig('matrice_confuzie_heatmap.png', dpi=300, bbox_inches='tight')

print(">>> Graficul matrice_confuzie_heatmap.png a fost salvat!")

plt.show()

print("\n" + "=" * 70)
print("REZOLVARE COMPLETA!")
print("=" * 70)
print("\nFisiere generate:")
print("  1. Cerinta1.csv - Anul cu cei mai multi angajati pe localitate")
print("  2. Cerinta2.csv - Rata ocuparii pe judete (sortata descrescator)")
print("  3. z.csv - Scoruri discriminante")
print("  4. matc.csv - Matricea de confuzie")
print("  5. grafic_scoruri_discriminante.png - Grafic 2D scoruri")
print("  6. predictii_pacienti.csv - Predictii pentru pacienti noi (BONUS)")
print("  7. matrice_confuzie_heatmap.png - Vizualizare matricea (BONUS)")
print("\nRezultate afisate la consola:")
print("  - Indicatori de acuratete (B.3)")
print("  - Numar componente principale semnificative (C)")
print("\nSUCCES LA EXAMEN!")
print("=" * 70)