import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

X_train = pd.read_csv('X_train.csv')
X_test = pd.read_csv('X_test.csv')
y_train = pd.read_csv('y_train.csv').squeeze()
y_test = pd.read_csv('y_test.csv').squeeze() # se usa el squeeza para pasarle un dataframe en lugar de una serie

print('Datos cargados:', X_train.shape, X_test.shape)

# Modelo de Decision
arbol = DecisionTreeClassifier(max_depth=4, random_state=42)
arbol.fit(X_train, y_train)

print('Arbol entrenado. Profundidad maxima configurada', 4)

# Comparar los modelos
logistica = LogisticRegression(max_iter=1000)
logistica.fit(X_train, y_train)

acc_logistica = accuracy_score(y_test, logistica.predict(X_test))
acc_arbol = accuracy_score(y_test, arbol.predict(X_test))

print(f'Regresion Logistica - accuracy: {acc_logistica:.4f}')
print(f'Arbol de Decision - accuracy: {acc_arbol:.4f}')

# Importancia de las variables
importancias = pd.Series(arbol.feature_importances_, index = X_train.columns)
importancias = importancias.sort_values(ascending = False)

print('\nLas 5 variables mas importantes del modelo arbol para decir')
print(importancias.head(5).round(3))















































