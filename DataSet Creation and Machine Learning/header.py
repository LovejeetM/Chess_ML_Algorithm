import pandas as pd
import os

#For creating Header of the dataset

#opath= os.path.join('dataset/black')

outcsv= 'dataset/white/white.csv'

#df = pd.read_csv(outcsv)
pixels1= 1200
columns = ["pixel_" + str(i) for i in range(pixels1)] + ["label"]
df = pd.DataFrame(columns=columns)


df.to_csv(outcsv, index=False)