import pandas as pd
import numpy as np


data = pd.read_csv('phone_data.csv')
df = pd.DataFrame(data)


print(df.head())