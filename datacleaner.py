import pandas as pd
import numpy as np


df  = pd.read_pickle("files/saved_data.pickle")
#for i in range(len(df)):
#    for elem in (df["landm arks"][i]):
#        df["landmarks"][i] = np.asarray(df["landmarks"][i])
#df.to_pickle("files/saved_data.pickle")
#df = df.drop([141])
#print(df["value"][130:])

for i in range(len(df)):
    if df["value"][i] == 'A':df["num_value"][i]=1
    if df["value"][i] == 'B':df["num_value"][i]=2
    if df["value"][i] == 'C':df["num_value"][i]=3
    if df["value"][i] == 'D':df["num_value"][i]=4
    if df["value"][i] == 'E':df["num_value"][i]=5
    if df["value"][i] == 'F':df["num_value"][i]=6
    if df["value"][i] == 'G':df["num_value"][i]=7
    if df["value"][i] == 'H':df["num_value"][i]=8
    if df["value"][i] == 'I':df["num_value"][i]=9
    if df["value"][i] == 'J':df["num_value"][i]=10
    if df["value"][i] == 'K':df["num_value"][i]=11
    if df["value"][i] == 'L':df["num_value"][i]=12
    if df["value"][i] == 'M':df["num_value"][i]=13
    if df["value"][i] == 'N':df["num_value"][i]=14
    if df["value"][i] == 'O':df["num_value"][i]=15
    if df["value"][i] == 'P':df["num_value"][i]=16
    if df["value"][i] == 'Q':df["num_value"][i]=17
    if df["value"][i] == 'R':df["num_value"][i]=18
    if df["value"][i] == 'S':df["num_value"][i]=19
    if df["value"][i] == 'T':df["num_value"][i]=20
    if df["value"][i] == 'U':df["num_value"][i]=21
    if df["value"][i] == 'V':df["num_value"][i]=22
    if df["value"][i] == 'W':df["num_value"][i]=23
    if df["value"][i] == 'X':df["num_value"][i]=24
    if df["value"][i] == 'Y':df["num_value"][i]=25
    if df["value"][i] == 'Z':df["num_value"][i]=26
df.to_pickle("files/saved_data.pickle")
