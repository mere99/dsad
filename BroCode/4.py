import pandas as pd

df = pd.read_csv("file.csv", index_col="Name")
# print(df)

#selection by COLUMNS
print(df[["Weight", "Height"]])
print("-----------------")
#selection by ROWS
#EACH ROW HAS A LABEL

print(df.loc["Kakuna", ["Height"]])
print("-----------------")
print(df.iloc[0:3]) #imi da randurile cu indicele 0,1 si 2, FARA 3!

