import pandas as pd
import numpy as np
import glob2
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')
from sklearn import linear_model
from sklearn.model_selection import train_test_split

class DataModel():

    def __init__(self):
        pass

    #Metodo encargado de ejecutar el proceso ETL y el algoritmo de prediccion
    def predictive_algorithm(self):

        resultado_final = False

        #Para el manejo de errores se implementó el metodo try-except
        try:
            #Se cargan los archivos que contienen los datos de entrada para nuestro modelo y se concatenan en un solo
            #dataframe
            path = 'D:\predictive_algorithm\src\data\input_data_*'
            all_files = glob2.glob(path)

            input_data = pd.DataFrame()

            for i in all_files:
                data_txt = pd.read_csv(i, index_col= None, header=None, sep = "|" ,names=["Col1"])
                input_data = input_data.append(data_txt)

            #Se debe estructurar la información para poder ser procesada por el algoritmo de prediccion
            #Se separan los datos por el delimitador :, se reemplazan los espacios en blanco por N/A para posteriormente ser filtrados
            complete_input_data = input_data["Col1"].str.rpartition(":", expand = True).replace(r'^\s*$', np.nan, regex=True).ffill(axis=0).dropna()

            #Se separan los datos por el delimitador ","
            complete_input_data = complete_input_data.rename(columns={0: "MovieID", 1: "Delimiter", 2: "Information"}).drop(columns=["Delimiter"])
            complete_input_data[["CustomerID","Rating","Date"]] = complete_input_data["Information"].str.split(",", expand = True)
            complete_input_data = complete_input_data.drop(columns=["Information"])

            #El algoritmo de prediccion solo funciona con valores númericos, se realiza casteo
            complete_input_data["MovieID"] = pd.to_numeric(complete_input_data['MovieID'])
            complete_input_data["CustomerID"] = pd.to_numeric(complete_input_data['CustomerID'])
            complete_input_data["Rating"] = pd.to_numeric(complete_input_data['Rating'])
            complete_input_data["Date"] = pd.to_datetime(complete_input_data['Date'])

            #Acotamos nuestros datos para mejorar la precisión del modelo, solo tendremos en consideración los primeros 150 usuarios
            complete_input_data = complete_input_data.loc[complete_input_data["CustomerID"] <= 150]
            complete_input_data.reset_index(drop=True, inplace=True)

            #Se filtran posibles valores duplicados
            complete_input_data = complete_input_data.drop_duplicates()

            #Cargar archivo que contiene la descripción de cada una de las peliculas, eliminar columnas que no utilizaremos y eliminar registros vacíos
            movie_description = pd.read_csv('D:/predictive_algorithm/src/data/movie_titles.csv', names=["MovieID","YearOfRelease","Title"]).dropna()
            movie_description = movie_description.drop(columns=["YearOfRelease"])

            #Para nuestro modelo vamos a utilizar el algoritmo de regresión logistica
            #Variables de entrada
            x_var = complete_input_data[["MovieID","CustomerID"]]
            #Variable objetivo
            y_var = complete_input_data["Rating"]

            #Este metodo se encarga de dividir nuestro dataset en datos de entranamiento y datos de prueba en una porporción de 75-25
            x_train,x_test,y_train,y_test = train_test_split(x_var, y_var, test_size=0.25)

            #Creamos el objeto de Regresión Lineal
            logist = linear_model.LogisticRegression(solver='liblinear')

            #Entrenamos el modelo
            logist.fit(x_train, y_train)

            #Realizamos las predicciones con nuestros datos de prueba
            rating = logist.predict(x_test)

            #Creamos dataframe para almacenar el resultado final
            resultado = pd.DataFrame(columns=["MovieID","CustomerID","Predicted_Rating"])

            x_test.reset_index(drop=True, inplace=True)

            for i in x_test.index:
                resultado = resultado.append({"MovieID":x_test.iloc[i,0], "CustomerID":x_test.iloc[i,1], "Predicted_Rating":rating[i]}, ignore_index=True)

            #Realizamos un left join entre el resultado obtenido y la descripcion de las peliculas
            resultado_final = resultado.merge(movie_description, on='MovieID', how = 'left')


        except Exception as e:
            log = open("D:/predictive_algorithm/src/data/log.txt", "a")
            log.write('\n' + "Error del modelo de prediccion: " + str(e))
            log.close()
            resultado_final = False

        finally:
            return resultado_final