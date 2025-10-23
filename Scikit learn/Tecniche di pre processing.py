import numpy as np
import pandas as pd 
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


#1 Creiamo un dataset con scale diverse
data = {
    'feature_A' : [1,2,3,4,5,100], #Outlier 100
    'feature_B' : [100,200,300,400,500,600] #scala grande ma regolare 
}

df = pd.DataFrame(data)
print("Dataset originale")
print(df)


#2 Standard Scaler

scaler_std =StandardScaler()
df_std = pd.DataFrame(scaler_std.fit_transform(df), columns=df.columns)

print ("Standard scaler (Z-Score)")
print (df_std)
print("Media: ", df_std.mean().round(2).to_dict())
print("Deviazione standard: ", df_std.std(ddof=0).round(2).to_dict())


scaler_mm = MinMaxScaler(feature_range=(0,1))
df_mm = pd.DataFrame(scaler_mm.fit_transform(df), columns=df.columns)

print("Min max Scaler [0,1]")
print(df_mm)
print("Min: ", df_mm.min().to_dict(), "Max: ", df_mm.max().to_dict())

scaler_rb = RobustScaler()
df_rb = pd.DataFrame(scaler_rb.fit_transform(df), columns=df.columns)

print ("Robust Scaler")
print(df_rb)
print("Mediana psot-scaling:", df_rb.median().round(2).to_dict())