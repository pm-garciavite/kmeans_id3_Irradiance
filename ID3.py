# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 16:35:03 2021

@author: Angel Zumaya
"""
import pandas
from sklearn import tree
import pydotplus
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import matplotlib.image as pltimg

# datos=pandas.read_csv("ID3D.csv")
datos=pandas.read_csv("ID3.csv")
#print(datos)
car=["Irradiancia","Temperatura","Potencia"]
X=datos[car]
Y=datos["label"]
#print(X)
#print("------")
#print(Y)

dtree=DecisionTreeClassifier()
dtree=dtree.fit(X,Y)
datos=tree.export_graphviz(dtree,out_file=None,feature_names=car)
graph=pydotplus.graph_from_dot_data(datos)
graph.write_png('miarbol.png')

imagen=pltimg.imread('miarbol.png')
implot=plt.imshow(imagen)
#plt.show()

dtree=DecisionTreeClassifier()
dtree=dtree.fit(X,Y)

#DATOS INGRESADOS POR EL USUARIO:
print("Predicción con ID3")
i=int(input("Ingresa la Irradiancia:"))
t=int(input("Ingresa la Temperatura:"))
p=float(input("Ingresa la Potencia:"))
#i=680 #irradiancia
#t=40 #temperatura
#p=0.29 #potencia

print("------")
print("Prediccion con",i, "de irradiancia, ",t,"de temperatura y ",p,"de potencia")
print("Grupo Resultante :",dtree.predict([[i,t,p]]))
print("0 = Potencia de salida baja, diodos bypass activados")
print("1 = Buena potencia de salida, diodos bypass desactivados")

plt.show()