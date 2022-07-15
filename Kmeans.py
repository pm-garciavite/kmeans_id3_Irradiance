# -*- coding: utf-8 -*-
"""
@author: arach
"""
from pandas import ExcelWriter
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn import tree
# import pydotplus
import matplotlib.image as pltimg
import matplotlib
matplotlib.rcParams['text.usetex'] = True

#Llamar al conjunto de datos G,T,P
data=pd.read_csv('datosit.csv')
X=data['Irradiancia']
Y=data['Temperatura']
Z=data['Potencia']
 
#Crear la figura 
fig = plt.figure(figsize = (10, 7))
ax = plt.axes(projection ="3d")
 
#Graficar los datos
ax.scatter3D(X, Y, Z, color = "blue")
plt.title("Datos", fontweight ='bold', fontsize=30)
ax.set_xlabel(r'Irradiancia (W/m$^2$)', fontweight ='bold', fontsize=20)
ax.set_ylabel('Temperatura (°C)', fontweight ='bold', fontsize=20)
ax.set_zlabel('Potencia (W)', fontweight ='bold', fontsize=20)
fig.savefig('datos.png', format='png', dpi=1000)
 
# show plot
plt.show()
#K-Means
kmeans = KMeans(n_clusters=2).fit(data)
centroids = kmeans.cluster_centers_
labels = kmeans.predict(data)
#Obtener los centroides
C = kmeans.cluster_centers_
colores=['green','purple']
asignar=[]
colors=[]
#Ciclo for para colorear datos correspondientes al centroide
for row in labels:
    asignar.append(colores[row])
#Inicia la figura 
fig1 = plt.figure(figsize = (10, 7))
ax1 = plt.axes(projection ="3d")
#Grafica los datos ya coloreados por separado 
ax1.scatter3D(X, Y, Z,  c=asignar,s=50)
#Grafica los centroides con color y marcador
ax1.scatter(C[:, 0], C[:, 1], C[:, 2], marker='o', c='black', s=100)
ax1.text(C[0, 0], C[0, 1]-1, C[0, 2], r'C$_1$', color='black', size=20)
ax1.text(C[1, 0], C[1, 1]-1, C[1, 2], r'C$_2$', color='black', size=20)
plt.title("Datos clasificados con K-Means", fontweight ='bold', fontsize=30)
ax1.set_xlabel(r'Irradiancia (W/m$^2$)', fontweight ='bold', fontsize=14)
ax1.set_ylabel('Temperatura (°C)', fontweight ='bold', fontsize=14)
ax1.set_zlabel('Potencia (W)', fontweight ='bold', fontsize=14)

plt.show()
#la mejor  la vista según
#ax1.view_init(elev=18, azim=-148)
ax1.view_init(elev=20, azim=-114)
print('Los centroides están ubicados en: ')
print(centroids)
fig1.savefig('kmeans.eps', format='eps', dpi=1000)
fig1.savefig('kmeans.png', format='png', dpi=1000)

# graph=pydotplus.graph_from_dot_data(datos)
# graph.write_png('miarbol.png')
# imagen=pltimg.imread('miarbol.png')
# implot=plt.imshow(imagen)


#Mostrar los grupos en los datos
data['label']= labels
print(data)

#Generar archivo .xlsx con los datos por grupos
p1=pd.DataFrame(data)
with pd.ExcelWriter('ID3.xlsx') as writer:
        p1.to_excel(writer)
