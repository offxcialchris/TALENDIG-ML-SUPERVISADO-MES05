import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import lineStyles

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, roc_auc_score

os.makedirs("graficas", exist_ok=True)

X_train = pd.read_csv('X_train.csv')
X_test = pd.read_csv('X_test.csv')
y_train = pd.read_csv('y_train.csv').squeeze()
y_test = pd.read_csv('y_test.csv').squeeze() # se usa el squeeza para pasarle un dataframe en lugar de una serie

# Modelo
modelo = LogisticRegression(max_iter=1000)
modelo.fit(X_train, y_train)

#Predicciones y probabilidades
y_pred = modelo.predict(X_test)
y_proba = modelo.predict_proba(X_test)[:, 1]

print("Modelo listo. Predicciones generadas:", len(y_test), "Pasajeros de prueba")

# Accuracy - prediccion que siempre salga "no sobrevivio"
prediccion_tonta = [0] * len(y_test)

accuracy_tonta = accuracy_score(y_test, prediccion_tonta)
accuracy_modelo = accuracy_score(y_test, y_pred)

print(f"Accuracy de un modelo que siempre predice 'No sobrevivio: {accuracy_tonta:.4f}' ")
print(f"Accuracy de nuestro modelo real: {accuracy_modelo:.4f}")

# Matriz de confusion
matriz = confusion_matrix(y_test, y_pred)
print('\nMatriz de confusion:')
print(matriz)
print(f'\nVerdadero Negativo (VN): {matriz[0][0]}, predijo no sobrevivio y acerto')
print(f'Falso Positivo (FP): {matriz[0][1]}, predijo sobrevivio pero so sobrevivio')
print(f'Falso Negativo (FN): {matriz[1][0]}, predijo no sobrevivio pero si sobrevivio')
print(f'Verdadero Positivo (VP): {matriz[1][1]}, predijo sobrevivio y acerto')


# Tarea foto de una hoja

# (vn + vp) / (vn + fp + fn + vp) |

# Visualizacion de matriz de correlacion
plt.figure(figsize = (6,5))
sns.heatmap(matriz, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No sobrevivio', 'Sobrevivio'],
            yticklabels=['No sobrevivio', 'Sobrevivio']) # argumento annot=True le dice a seaborn que escriba el numero dentro de cada celda, el fmt le especifica a seaborn que ensenie numero decimal a 90
plt.title('Matriz de Confusion')
plt.xlabel('Prediccion de modelo')
plt.ylabel('Valor')
plt.tight_layout()
plt.savefig('../graficas/02_matriz_confusion.png')
plt.show()

# Formulas

# Precision = VP / (VP + FP)
# Recall = VP / (VP + FN)
# F1 = 2 * (Precision * Recall) / (Precision + Recall)

# Metricas Precision, Recall, F1
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f'Precison: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1: {f1:.4f}')

# Metrica classificacion_report
print(classification_report(y_test, y_pred, target_names=['No sobrevivio', 'Sobrevivio']))

# Curva ROC y AUX
fpr, tpr, umbrales = roc_curve(y_test, y_proba) # false positive rate(mas bajo mejor), true positive rate=recall
auc = roc_auc_score(y_test, y_proba)

plt.figure(figsize = (7,6))
plt.plot(fpr, tpr, color='steelblue', label=f'Modelo (AUC = {auc:.3f})')
plt.plot([0,1], [0,1], color='gray', linestyle='--', label='Modelo al azar (AUC = 0.5)') # linea diagonal, adivinacion al azar
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos (Recall)')
plt.legend()
plt.tight_layout()
plt.savefig('../graficas/03_roc_curve.png')
plt.show()

print(f'AUC (AREA BAJO LA CURVA): {auc:.4f}')

























































