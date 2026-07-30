import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns

os.makedirs("graficas", exist_ok=True)

df = pd.read_csv('housing.csv')

print('Forma del dataset:', df.shape)
print('\nColumnas disponibles:')
print(list(df.columns))
print('\nPrimeras filas:')
print(df.head())
print('\nTipos de datos:')
print(df.dtypes)

print(df.isnull().sum())
print()
print(df['ocean_proximity'].value_counts())

# modelo regresion lineal simple

# Se agregan doble brackets porque
# scikit-learn siempre pedira un dataframe y no una columna.
# Truceando el modelo para que crea que la columna es un DataFrame.
# Porque se usa DataFrame para entrenar modelos
# modelo de regresion en X si es una sola columna
# se usaran dobles brackets para truquear el modelo
# y que piense que es un data frame.
# modelo de regresion en "y", se usara un solo bracket
# porque sera solo esa columna que se intentara predecir

X_simple = df[['median_income']]
y = df['median_house_value']

modelo_simple = LinearRegression()
modelo_simple.fit(X_simple, y)

# la pendiente representa el cambio o incremento en la
# varaiable que queremos predecir
print('Pendiente:', round(modelo_simple.coef_[0], 2))
# El intercepto es simplemetne el punto donde
# la recta cruza el eje en Y
print('Intercepto:', round(modelo_simple.intercept_, 2))

ingreso_nuevo = [[8.0]]
prediccion = modelo_simple.predict(ingreso_nuevo)

print(f"Para un ingreso medio de 8.0, el modelo predice: ${prediccion[0]}:,.2f")

# Visualizacion
plt.figure(figsize=(9, 6))

plt.scatter(
    df['median_income'],
    df['median_house_value'],
    alpha=0.1,
    color='steelblue'
)

plt.plot(
    df['median_income'],
    modelo_simple.predict(X_simple),
    color='red',
    linewidth=2
)

plt.title('Regresión lineal simple: Ingreso vs Valor de la vivienda')
plt.xlabel('Ingreso medio')
plt.ylabel('Valor medio de la vivienda')

plt.tight_layout()
plt.savefig('graficas/01_regresion_simple.png')
plt.show()


















