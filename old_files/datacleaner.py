import pandas as pd
import numpy as np


df  = pd.read_pickle("files/saved_data.pickle")
print(df)

for i in range(len(df)):
    df["landmarks"][i] = np.asarray(df["landmarks"][i])

#for i in range(len(df)):
#    for elem in (df["landmarks"][i]):
#        df["landmarks"][i] = np.asarray(df["landmarks"][i])
df.to_pickle("files/saved_data.pickle")


print(df)

#datacleaner on number 6 -> set value to G instead of H

