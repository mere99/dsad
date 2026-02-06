import pandas as pd

#DataFrame = un tabel cu linii si coloane; similar to an Excel spreadsheet

data={"name":["angela","rick","joy"],
      "age":[39,40,28]
      }

#constructor prin care transf dictionary in df
df = pd.DataFrame(data, index=["employee 1", "employee 2", "employee 3"])

print(df.loc["employee 1"]) #selecting a specific row by label
print(df.iloc[1])#selecting by index
print(df)
print("-------------------")

#add a new column
df["job"] =["cook", "intern", "cashier"]

#add a new ROW - create a new df with a single row, then we concatenate
new_row = pd.DataFrame([{"name": "sandy", "age":28, "job":"scientist"}],
                       index=["employee 4"])
df = pd.concat([df, new_row])

print(df)
