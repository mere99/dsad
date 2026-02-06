import pandas as pd

data = [100,102,104,201, 202]

#Series e ca o coloana din Excel
series = pd.Series(data, index=["a", "b", "c", "d", "e"])
series.loc["c"]=200 #loc = location by label

print(series.iloc[0]) #iloc= location by index
print(series.iloc[1])
print(series.iloc[2])
print(series)

print(series[series>=200])