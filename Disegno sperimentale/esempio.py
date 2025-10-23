import numpy as np

np.random.seed(0)

n = 20
gruppi = np.random.choice(['Farmaco','Placebo'], size=n)
print(gruppi)

fertilizzante_A = [4.1,4.3,4.0,4.2]
fertilizzante_B = [5.0 , 4.8 , 5.2 , 4.9]

print("Media A:", np.mean(fertilizzante_A))
print("Deviazione standard A:", np.std(fertilizzante_A, ddof=1))

print("Media B:", np.mean(fertilizzante_B))
print("Deviazione standard B:", np.std(fertilizzante_B, ddof=1))