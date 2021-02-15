from src.model.DataModel import DataModel
import pandas as pd

pd.set_option("display.max_rows", None, "display.max_columns", None)

#Instanciamos un objeto de tipo Datamodel y ejecutamos el método predictive_algorithm
resultado = DataModel().predictive_algorithm()

print(resultado)