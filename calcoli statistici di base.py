import numpy as np
from scipy import stats

x = [4,4,4,6,6,8,10,12]
mean = np.mean(x)
median = np.median(x)
mode = stats.mode(x, keepdims=True)
variance = np.var(x,ddof=1) # Varianza su campione serve DDOF1 per l'ottimizzazione di Bessel
std_dev = np.std(x,ddof=1) 
iqr_value = stats.iqr(x)

print("Media:", mean)
print("Mediana:", median)
print("Moda:", mode.mode[0])
print("Varianza", variance)
print("Deviazione standard:", std_dev)
print("Scarto interquartile:", iqr_value)

