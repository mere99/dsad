#filtering = keeping the rows that match a condition
import pandas as pd

df = pd.read_csv("file.csv")

tall_pokemon=df[df["Height"] >= 2]
print(tall_pokemon)
print("---------")

heavy_pokemon = df[df["Weight"] >100]
print(heavy_pokemon)
print("---------")

water_pokemon = df[(df["Type1"] =="Water")|
                   (df["Type2"] =="Water")]
print(water_pokemon)
