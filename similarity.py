import os
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
from os.path import join


curr_dir = os.getcwd()
data = join(curr_dir, "all_data.txt")


d = np.loadtxt(data)


df = pd.DataFrame(d, columns = ['1', '2', '3', '4', '5', '6', '7', 'labels'])
df['labels'] = df['labels'].round(decimals = 1) #round off labels to 6 decimal places
print(df.labels.unique())
print(df['labels'].value_counts())
duplicate = df[df.duplicated(['1', '2', '3', '4', '5','6','7', 'labels'])]


with pd.option_context('display.max_rows', None):
    print(duplicate)
   
print(len(duplicate))





