import numpy as np
import pandas as pd



def media_evolve(lista_datos: list):
    # Calcular la media sumando los datos y dividiendo por la cantidad de datos
    return sum(lista_datos) / len(lista_datos)

def mediana_evolve(lista_datos: list):
    # Ordenar los datos y calcular la mediana
    datos_ordenados = sorted(lista_datos)
    n = len(datos_ordenados)
    if n % 2 == 0:
        return (datos_ordenados[n // 2 - 1] + datos_ordenados[n // 2]) / 2
    else:
        return datos_ordenados[n // 2]

def percentil_evolve(lista_datos: list, percentil: int):
    # Ordenar los datos y calcular el percentil utilizando la fórmula de interpolación  
    datos_ordenados = sorted(lista_datos)
    n = len(datos_ordenados)
    idx = (percentil / 100) * (n - 1)
    lower = int(idx)
    upper = lower + 1
    if upper >= n:
        return datos_ordenados[lower]
    fraction = idx - lower
    return datos_ordenados[lower] + fraction * (datos_ordenados[upper] - datos_ordenados[lower])

def varianza_evolve(lista_datos: list):
    # Calcular la varianza utilizando la media y la fórmula de varianza
    media = media_evolve(lista_datos)
    return sum((x - media) ** 2 for x in lista_datos) / len(lista_datos)

def desviacion_evolve(lista_datos: list):
    # Calcular la desviación estándar como la raíz cuadrada de la varianza  
    return varianza_evolve(lista_datos) ** 0.5

def IQR_evolve(lista_datos: list):
    # Calcular el rango intercuartílico (IQR) como la diferencia entre el tercer y primer cuartil   
    q1 = percentil_evolve(lista_datos, 25)
    q3 = percentil_evolve(lista_datos, 75)
    return q3 - q1




if __name__ == "__main__":
    # Generar datos de ejemplo para edad, salario y experiencia utilizando numpy
    np.random.seed(42)
    edad = list(np.random.randint(20, 60, 100))
    salario =  list(np.random.normal(45000, 15000, 100))
    experiencia = list(np.random.randint(0, 30, 100))

    np.random.seed(42)
    # Crear un DataFrame de pandas con los mismos datos para comparar los resultados
    df = pd.DataFrame({
        'edad': np.random.randint(20, 60, 100),
        'salario': np.random.normal(45000, 15000, 100),
        'experiencia': np.random.randint(0, 30, 100)
    })
    
    print("Resultado pandas:")
    print("-----------------------------")
    print(df.describe())

    print("Resultado edad:")
    print("-----------------------------")
    print(media_evolve(edad))
    print(mediana_evolve(edad))
    print(percentil_evolve(edad, 50))
    print(varianza_evolve(edad))
    print(desviacion_evolve(edad))
    print(IQR_evolve(edad))

    print(media_evolve(salario))
    print(mediana_evolve(salario))
    print(percentil_evolve(salario, 50))
    print(varianza_evolve(salario))
    print(desviacion_evolve(salario))
    print(IQR_evolve(salario))

    print(media_evolve(experiencia))
    print(mediana_evolve(experiencia))
    print(percentil_evolve(experiencia, 50))
    print(varianza_evolve(experiencia))
    print(desviacion_evolve(experiencia))
    print(IQR_evolve(experiencia))