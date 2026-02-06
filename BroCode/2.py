import pandas as pd

#dictionary
calories = {"day1": 1750, "day2":2100, "day3":1700}

series = pd.Series(calories)
print(series)

series.loc["day3"] += 500
print(series.loc["day3"])
print(series[series>2000])