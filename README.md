Sub A (2 cerințe) - Manipulare date + agregări
Sub B (3 cerințe) - Metodă principală de analiză
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
