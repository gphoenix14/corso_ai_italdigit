from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Carica il dataset
digits = datasets.load_digits()
X, y = digits.data, digits.target

# Limitare a una classificazione binaria (ad esempio, cifre 0 e 1)
X = X[y < 2]
y = y[y < 2]

# Suddivide il dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Inizializza il modello
svm = SVC(kernel='linear')

# Allena il modello
svm.fit(X_train, y_train)

# Effettua previsioni
y_pred = svm.predict(X_test)

# Valuta l'accuratezza
print("Accuratezza:", accuracy_score(y_test, y_pred))
