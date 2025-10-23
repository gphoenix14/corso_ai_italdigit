from sklearn.datasets import load_iris 
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# 1 Caricamento del dataset
dataset = load_wine()
x = dataset.data 
y = dataset.target

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

knn = KNeighborsClassifier(n_neighbors=20)

knn.fit(X_train,y_train)

y_pred = knn.predict(X_test)

print("Accuratezza:", accuracy_score(y_test, y_pred))
print("\nReport di classificazione:\n", classification_report(y_test,y_pred, target_names=dataset.target_names))
