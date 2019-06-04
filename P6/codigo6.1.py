import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
from scipy.spatial import distance
import networkx as nx
import math


measurements = [ 'PM10', 'valid']
MINIMUM = 30

d = pd.read_csv('filled.csv', dtype = {'station': str, 'notes': str, 'timestamp': str})
d.station = d.station.astype('category')
d.timestamp = pd.to_datetime(d.timestamp, format='%d-%b-%y %H')

#####   SERNA   #####
select_indices = list(np.where(d["timestamp"] == '2017-10-01 00:00:00')[0])
indice = select_indices[0]
d = d.iloc[indice:]

select_last_indices = list(np.where(d["timestamp"] == '2017-10-31 23:00:00')[0])
last_indice = select_last_indices[0]
d = d.iloc[:last_indice]

d = d.drop(["CO", "PM2_5", "NO2", "NOX", "O3", "NO", "pressure",
            "rainfall", "humidity", "SO2", "solar", "temperature", "notes"], axis=1)

por_estacion = []

stat = d.groupby('station')
for s in d.station.cat.categories:
    sd = stat.get_group(s)
    por_estacion.append(sd)
    
centro = por_estacion[0]
centro = centro.dropna()
result = sm.ols(formula="PM10 ~ velocity + direction", data=centro).fit()
respuesta_centro = result.params[0]

#DIBUJITO 
"""lista = []
for i in range(len(centro.PM10)):
    lista.append(i)

centro = centro.set_axis(lista, axis='index', inplace=False)
    
area = np.pi*(1)
#plt.plot(centro.index, centro['direction'], 'r--')
plt.scatter(centro.index, centro['direction'], s=area, c='b')
plt.xlim(0,720)
plt.title('Estación Centro')
plt.ylabel('PM10')
plt.xlabel('Noviembre de 2017')
plt.savefig('11_17.jpg', format='jpg', dpi=1000)"""



noreste = por_estacion[1]
noreste = noreste.dropna()
result = sm.ols(formula="PM10 ~ velocity + direction", data=noreste).fit()
respuesta_noreste = result.params[0]

noreste2 = por_estacion[2]
result = sm.ols(formula="PM10 ~ velocity", data=noreste2).fit()
respuesta_noreste2 = result.params[0]

noroeste = por_estacion[3]
noroeste = noroeste.dropna()
result = sm.ols(formula="PM10 ~ velocity + direction", data=noroeste).fit()
respuesta_noroeste = result.params[0]

noroeste2 = por_estacion[4]
noroeste2 = noroeste2.dropna()
result = sm.ols(formula="PM10 ~ velocity + direction", data=noroeste2).fit()
respuesta_noroeste2 = result.params[0]

norte = por_estacion[5]
norte = norte.dropna()
result = sm.ols(formula="PM10 ~ velocity + direction", data=norte).fit()
respuesta_norte = result.params[0]

norte2 = por_estacion[6]
norte2 = norte.dropna()
result = sm.ols(formula="PM10 ~ velocity + direction", data=norte2).fit()
respuesta_norte2 = result.params[0]

sur = por_estacion[7]
result = sm.ols(formula="PM10 ~ direction", data=sur).fit()
respuesta_sur = result.params[0]

sureste = por_estacion[8]
result = sm.ols(formula="PM10 ~ velocity", data=sureste).fit()
respuesta_sureste = result.params[0]

sureste2 = por_estacion[9]
sureste2 = sureste2.dropna()
result = sm.ols(formula="PM10 ~ velocity + direction", data=sureste2).fit()
respuesta_sureste2 = result.params[0]

sureste3 = por_estacion[10]
result = sm.ols(formula="PM10 ~ direction", data=sureste3).fit()
respuesta_sureste3 = result.params[0]

suroeste = por_estacion[11]
result = sm.ols(formula="PM10 ~ velocity", data=suroeste).fit()
respuesta_suroeste = result.params[0]

suroeste2 = por_estacion[12]
oresult = sm.ols(formula="PM10 ~ direction", data=suroeste2).fit()
respuesta_suroeste2 = result.params[0]



estimado_2017_11_01 = [respuesta_sureste, respuesta_noreste, respuesta_centro,
                       respuesta_noroeste, respuesta_suroeste, respuesta_noroeste2,
                       respuesta_norte, respuesta_noreste2, respuesta_sureste2,
                       respuesta_suroeste2, respuesta_sureste3,
                       respuesta_norte2, respuesta_sur]

d = pd.read_csv('filled.csv', dtype = {'station': str, 'notes': str, 'timestamp': str})
d.station = d.station.astype('category')
d.timestamp = pd.to_datetime(d.timestamp, format='%d-%b-%y %H')

real_2017_11_01_indices = np.where(d["timestamp"] == '2017-11-01 00:00:00')[0]

real_2017_11_01 = [66.0, 137.0, 0, 205.0, 104.0, 140.0, 72.0, 139.0, 0,
                   0, 113.0, 61.0, 109.0]
vel_2017_11_01 = [4.8, 0.9, 0, 0, 1.3, 2.6, 1.3, 1.4, 2.6, 0.6, 1.3,
                  2.0, 2.2]
dir_2017_11_01 = [0, 344.0, 0, 264.0, 0, 162.0, 247.0, 0, 296.0, 218.0,
                  358.0, 234.0, 341.0]

x1 = sum(real_2017_11_01) / float(len(real_2017_11_01))
x2 = sum(vel_2017_11_01) / float(len(vel_2017_11_01))
x3 = sum(dir_2017_11_01) / float(len(dir_2017_11_01))


real = [66.0, 137.0, 88.15384615384616, 205.0, 104.0, 140.0, 72.0,
                   139., 88.15384615384616, 88.15384615384616,
                   113.0, 61.0, 109.0]
velocidad = [4.8, 0.9, 1.6153846153846154, 1.6153846153846154, 1.3, 2.6,
                  1.3, 1.4, 2.6, 0.6, 1.3,2.0, 2.2]
direction = [189.53846153846155, 344.0, 189.53846153846155, 264.0,
                  189.53846153846155, 162.0, 247.0, 189.53846153846155,
                  296.0, 218.0, 358.0, 234.0, 341.0]


######################################
#  sistema de ecuaciones 2017-11-01  #
######################################

coor = pd.read_csv('C:/Users/Serna/Dropbox/2doSemestre/Redes/tarea6/coordenadas.csv')
coor.head()


distancia = coor.as_matrix()
A = ([[(0 * velocidad[0] + direction[0]), (1/(distance.euclidean(list(distancia[0]), list(distancia[1]))) * velocidad[1] + direction[1]), (1/(distance.euclidean(list(distancia[0]), list(distancia[2]))) * velocidad[2] + direction[2]), (1/(distance.euclidean(list(distancia[0]), list(distancia[3]))) * velocidad[3] + direction[3]), (1/(distance.euclidean(list(distancia[0]), list(distancia[4]))) * velocidad[4] + direction[4]), (1/(distance.euclidean(list(distancia[0]), list(distancia[5]))) * velocidad[5] + direction[5]), (1/(distance.euclidean(list(distancia[0]), list(distancia[6]))) * velocidad[6] + direction[6]), (1/(distance.euclidean(list(distancia[0]), list(distancia[7]))) * velocidad[7] + direction[7]), (1/(distance.euclidean(list(distancia[0]), list(distancia[8]))) * velocidad[8] + direction[8]), (1/(distance.euclidean(list(distancia[0]), list(distancia[9]))) * velocidad[9] + direction[9]), (1/(distance.euclidean(list(distancia[0]), list(distancia[10]))) * velocidad[10] + direction[10]), (1/(distance.euclidean(list(distancia[0]), list(distancia[11]))) * velocidad[11] + direction[11]), (1/(distance.euclidean(list(distancia[0]), list(distancia[12]))) * velocidad[11] + direction[11])],
       [(1/(distance.euclidean(list(distancia[1]), list(distancia[0]))) * velocidad[0] + direction[0]), (0 * velocidad[1] + direction[1]), (1/(distance.euclidean(list(distancia[1]), list(distancia[2]))) * velocidad[2] + direction[2]), (1/(distance.euclidean(list(distancia[1]), list(distancia[3]))) * velocidad[3] + direction[3]), (1/(distance.euclidean(list(distancia[1]), list(distancia[4]))) * velocidad[4] + direction[4]), (1/(distance.euclidean(list(distancia[1]), list(distancia[5]))) * velocidad[5] + direction[5]), (1/(distance.euclidean(list(distancia[1]), list(distancia[6]))) * velocidad[6] + direction[6]), (1/(distance.euclidean(list(distancia[1]), list(distancia[7]))) * velocidad[7] + direction[7]), (1/(distance.euclidean(list(distancia[1]), list(distancia[8]))) * velocidad[8] + direction[8]), (1/(distance.euclidean(list(distancia[1]), list(distancia[9]))) * velocidad[9] + direction[9]), (1/(distance.euclidean(list(distancia[1]), list(distancia[10]))) * velocidad[10] + direction[10]), (1/(distance.euclidean(list(distancia[1]), list(distancia[11]))) * velocidad[11] + direction[11]), (1/(distance.euclidean(list(distancia[1]), list(distancia[12]))) * velocidad[11] + direction[11])],
       [(1/(distance.euclidean(list(distancia[2]), list(distancia[0]))) * velocidad[0] + direction[0]), (1/(distance.euclidean(list(distancia[2]), list(distancia[1]))) * velocidad[1] + direction[1]), (0 * velocidad[2] + direction[2]), (1/(distance.euclidean(list(distancia[2]), list(distancia[3]))) * velocidad[3] + direction[3]), (1/(distance.euclidean(list(distancia[2]), list(distancia[4]))) * velocidad[4] + direction[4]), (1/(distance.euclidean(list(distancia[2]), list(distancia[5]))) * velocidad[5] + direction[5]), (1/(distance.euclidean(list(distancia[2]), list(distancia[6]))) * velocidad[6] + direction[6]), (1/(distance.euclidean(list(distancia[2]), list(distancia[7]))) * velocidad[7] + direction[7]), (1/(distance.euclidean(list(distancia[2]), list(distancia[8]))) * velocidad[8] + direction[8]), (1/(distance.euclidean(list(distancia[2]), list(distancia[9]))) * velocidad[9] + direction[9]), (1/(distance.euclidean(list(distancia[2]), list(distancia[10]))) * velocidad[10] + direction[10]), (1/(distance.euclidean(list(distancia[2]), list(distancia[11]))) * velocidad[11] + direction[11]), (1/(distance.euclidean(list(distancia[2]), list(distancia[12]))) * velocidad[11] + direction[11])],
       [(1/(distance.euclidean(list(distancia[3]), list(distancia[0]))) * velocidad[0] + direction[0]), (1/(distance.euclidean(list(distancia[3]), list(distancia[1]))) * velocidad[1] + direction[1]), (1/(distance.euclidean(list(distancia[3]), list(distancia[2]))) * velocidad[2] + direction[2]), (0 * velocidad[3] + direction[3]), (1/(distance.euclidean(list(distancia[3]), list(distancia[4]))) * velocidad[4] + direction[4]), (1/(distance.euclidean(list(distancia[3]), list(distancia[5]))) * velocidad[5] + direction[5]), (1/(distance.euclidean(list(distancia[3]), list(distancia[6]))) * velocidad[6] + direction[6]), (1/(distance.euclidean(list(distancia[3]), list(distancia[7]))) * velocidad[7] + direction[7]), (1/(distance.euclidean(list(distancia[3]), list(distancia[8]))) * velocidad[8] + direction[8]), (1/(distance.euclidean(list(distancia[3]), list(distancia[9]))) * velocidad[9] + direction[9]), (1/(distance.euclidean(list(distancia[3]), list(distancia[10]))) * velocidad[10] + direction[10]), (1/(distance.euclidean(list(distancia[3]), list(distancia[11]))) * velocidad[11] + direction[11]), (1/(distance.euclidean(list(distancia[3]), list(distancia[12]))) * velocidad[11] + direction[11])],
       [(1/(distance.euclidean(list(distancia[4]), list(distancia[0]))) * velocidad[0] + direction[0]), (1/(distance.euclidean(list(distancia[4]), list(distancia[1]))) * velocidad[1] + direction[1]), (1/(distance.euclidean(list(distancia[4]), list(distancia[2]))) * velocidad[2] + direction[2]), (1/(distance.euclidean(list(distancia[4]), list(distancia[3]))) * velocidad[3] + direction[3]), (0 * velocidad[4] + direction[4]), (1/(distance.euclidean(list(distancia[4]), list(distancia[5]))) * velocidad[5] + direction[5]), (1/(distance.euclidean(list(distancia[4]), list(distancia[6]))) * velocidad[6] + direction[6]), (1/(distance.euclidean(list(distancia[4]), list(distancia[7]))) * velocidad[7] + direction[7]), (1/(distance.euclidean(list(distancia[4]), list(distancia[8]))) * velocidad[8] + direction[8]), (1/(distance.euclidean(list(distancia[4]), list(distancia[9]))) * velocidad[9] + direction[9]), (1/(distance.euclidean(list(distancia[4]), list(distancia[10]))) * velocidad[10] + direction[10]), (1/(distance.euclidean(list(distancia[4]), list(distancia[11]))) * velocidad[11] + direction[11]), (1/(distance.euclidean(list(distancia[4]), list(distancia[12]))) * velocidad[11] + direction[11])],
       [(1/(distance.euclidean(list(distancia[5]), list(distancia[0]))) * velocidad[0] + direction[0]), (1/(distance.euclidean(list(distancia[5]), list(distancia[1]))) * velocidad[1] + direction[1]), (1/(distance.euclidean(list(distancia[5]), list(distancia[2]))) * velocidad[2] + direction[2]), (1/(distance.euclidean(list(distancia[5]), list(distancia[3]))) * velocidad[3] + direction[3]), (1/(distance.euclidean(list(distancia[5]), list(distancia[4]))) * velocidad[4] + direction[4]), (0 * velocidad[5] + direction[5]), (1/(distance.euclidean(list(distancia[5]), list(distancia[6]))) * velocidad[6] + direction[6]), (1/(distance.euclidean(list(distancia[5]), list(distancia[7]))) * velocidad[7] + direction[7]), (1/(distance.euclidean(list(distancia[5]), list(distancia[8]))) * velocidad[8] + direction[8]), (1/(distance.euclidean(list(distancia[5]), list(distancia[9]))) * velocidad[9] + direction[9]), (1/(distance.euclidean(list(distancia[5]), list(distancia[10]))) * velocidad[10] + direction[10]), (1/(distance.euclidean(list(distancia[5]), list(distancia[11]))) * velocidad[11] + direction[11]), (1/(distance.euclidean(list(distancia[5]), list(distancia[12]))) * velocidad[11] + direction[11])],
       [(1/(distance.euclidean(list(distancia[6]), list(distancia[0]))) * velocidad[0] + direction[0]), (1/(distance.euclidean(list(distancia[6]), list(distancia[1]))) * velocidad[1] + direction[1]), (1/(distance.euclidean(list(distancia[6]), list(distancia[2]))) * velocidad[2] + direction[2]), (1/(distance.euclidean(list(distancia[6]), list(distancia[3]))) * velocidad[3] + direction[3]), (1/(distance.euclidean(list(distancia[6]), list(distancia[4]))) * velocidad[4] + direction[4]), (1/(distance.euclidean(list(distancia[6]), list(distancia[5]))) * velocidad[5] + direction[5]), (0 * velocidad[6] + direction[6]), (1/(distance.euclidean(list(distancia[6]), list(distancia[7]))) * velocidad[7] + direction[7]), (1/(distance.euclidean(list(distancia[6]), list(distancia[8]))) * velocidad[8] + direction[8]), (1/(distance.euclidean(list(distancia[6]), list(distancia[9]))) * velocidad[9] + direction[9]), (1/(distance.euclidean(list(distancia[6]), list(distancia[10]))) * velocidad[10] + direction[10]), (1/(distance.euclidean(list(distancia[6]), list(distancia[11]))) * velocidad[11] + direction[11]), (1/(distance.euclidean(list(distancia[6]), list(distancia[12]))) * velocidad[11] + direction[11])],
       [(1/(distance.euclidean(list(distancia[7]), list(distancia[0]))) * velocidad[0] + direction[0]), (1/(distance.euclidean(list(distancia[7]), list(distancia[1]))) * velocidad[1] + direction[1]), (1/(distance.euclidean(list(distancia[7]), list(distancia[2]))) * velocidad[2] + direction[2]), (1/(distance.euclidean(list(distancia[7]), list(distancia[3]))) * velocidad[3] + direction[3]), (1/(distance.euclidean(list(distancia[7]), list(distancia[4]))) * velocidad[4] + direction[4]), (1/(distance.euclidean(list(distancia[7]), list(distancia[5]))) * velocidad[5] + direction[5]), (1/(distance.euclidean(list(distancia[7]), list(distancia[6]))) * velocidad[6] + direction[6]), (0 * velocidad[7] + direction[7]), (1/(distance.euclidean(list(distancia[7]), list(distancia[8]))) * velocidad[8] + direction[8]), (1/(distance.euclidean(list(distancia[7]), list(distancia[9]))) * velocidad[9] + direction[9]), (1/(distance.euclidean(list(distancia[7]), list(distancia[10]))) * velocidad[10] + direction[10]), (1/(distance.euclidean(list(distancia[7]), list(distancia[11]))) * velocidad[11] + direction[11]), (1/(distance.euclidean(list(distancia[7]), list(distancia[12]))) * velocidad[11] + direction[11])],
       [(1/(distance.euclidean(list(distancia[8]), list(distancia[0]))) * velocidad[0] + direction[0]), (1/(distance.euclidean(list(distancia[8]), list(distancia[1]))) * velocidad[1] + direction[1]), (1/(distance.euclidean(list(distancia[8]), list(distancia[2]))) * velocidad[2] + direction[2]), (1/(distance.euclidean(list(distancia[8]), list(distancia[3]))) * velocidad[3] + direction[3]), (1/(distance.euclidean(list(distancia[8]), list(distancia[4]))) * velocidad[4] + direction[4]), (1/(distance.euclidean(list(distancia[8]), list(distancia[5]))) * velocidad[5] + direction[5]), (1/(distance.euclidean(list(distancia[8]), list(distancia[6]))) * velocidad[6] + direction[6]), (1/(distance.euclidean(list(distancia[8]), list(distancia[7]))) * velocidad[7] + direction[7]), (0 * velocidad[8] + direction[8]), (1/(distance.euclidean(list(distancia[8]), list(distancia[9]))) * velocidad[9] + direction[9]), (1/(distance.euclidean(list(distancia[8]), list(distancia[10]))) * velocidad[10] + direction[10]), (1/(distance.euclidean(list(distancia[8]), list(distancia[11]))) * velocidad[11] + direction[11]), (1/(distance.euclidean(list(distancia[8]), list(distancia[12]))) * velocidad[11] + direction[11])],
       [(1/(distance.euclidean(list(distancia[9]), list(distancia[0]))) * velocidad[0] + direction[0]), (1/(distance.euclidean(list(distancia[9]), list(distancia[1]))) * velocidad[1] + direction[1]), (1/(distance.euclidean(list(distancia[9]), list(distancia[2]))) * velocidad[2] + direction[2]), (1/(distance.euclidean(list(distancia[9]), list(distancia[3]))) * velocidad[3] + direction[3]), (1/(distance.euclidean(list(distancia[9]), list(distancia[4]))) * velocidad[4] + direction[4]), (1/(distance.euclidean(list(distancia[9]), list(distancia[5]))) * velocidad[5] + direction[5]), (1/(distance.euclidean(list(distancia[9]), list(distancia[6]))) * velocidad[6] + direction[6]), (1/(distance.euclidean(list(distancia[9]), list(distancia[7]))) * velocidad[7] + direction[7]), (1/(distance.euclidean(list(distancia[9]), list(distancia[8]))) * velocidad[8] + direction[8]), (0 * velocidad[9] + direction[9]), (1/(distance.euclidean(list(distancia[9]), list(distancia[10]))) * velocidad[10] + direction[10]), (1/(distance.euclidean(list(distancia[9]), list(distancia[11]))) * velocidad[11] + direction[11]), (1/(distance.euclidean(list(distancia[9]), list(distancia[12]))) * velocidad[11] + direction[11])],
       [(1/(distance.euclidean(list(distancia[10]), list(distancia[0]))) * velocidad[0] + direction[0]), (1/(distance.euclidean(list(distancia[10]), list(distancia[1]))) * velocidad[1] + direction[1]), (1/(distance.euclidean(list(distancia[10]), list(distancia[2]))) * velocidad[2] + direction[2]), (1/(distance.euclidean(list(distancia[10]), list(distancia[3]))) * velocidad[3] + direction[3]), (1/(distance.euclidean(list(distancia[10]), list(distancia[4]))) * velocidad[4] + direction[4]), (1/(distance.euclidean(list(distancia[10]), list(distancia[5]))) * velocidad[5] + direction[5]), (1/(distance.euclidean(list(distancia[10]), list(distancia[6]))) * velocidad[6] + direction[6]), (1/(distance.euclidean(list(distancia[0]), list(distancia[7]))) * velocidad[7] + direction[7]), (1/(distance.euclidean(list(distancia[10]), list(distancia[8]))) * velocidad[8] + direction[8]), (1/(distance.euclidean(list(distancia[10]), list(distancia[9]))) * velocidad[9] + direction[9]), (0 * velocidad[10] + direction[10]), (1/(distance.euclidean(list(distancia[10]), list(distancia[11]))) * velocidad[11] + direction[11]), (1/(distance.euclidean(list(distancia[10]), list(distancia[12]))) * velocidad[11] + direction[11])],
       [(1/(distance.euclidean(list(distancia[11]), list(distancia[0]))) * velocidad[0] + direction[0]), (1/(distance.euclidean(list(distancia[11]), list(distancia[1]))) * velocidad[1] + direction[1]), (1/(distance.euclidean(list(distancia[11]), list(distancia[2]))) * velocidad[2] + direction[2]), (1/(distance.euclidean(list(distancia[11]), list(distancia[3]))) * velocidad[3] + direction[3]), (1/(distance.euclidean(list(distancia[11]), list(distancia[4]))) * velocidad[4] + direction[4]), (1/(distance.euclidean(list(distancia[11]), list(distancia[5]))) * velocidad[5] + direction[5]), (1/(distance.euclidean(list(distancia[11]), list(distancia[6]))) * velocidad[6] + direction[6]), (1/(distance.euclidean(list(distancia[0]), list(distancia[7]))) * velocidad[7] + direction[7]), (1/(distance.euclidean(list(distancia[11]), list(distancia[8]))) * velocidad[8] + direction[8]), (1/(distance.euclidean(list(distancia[11]), list(distancia[9]))) * velocidad[9] + direction[9]), (1/(distance.euclidean(list(distancia[11]), list(distancia[10]))) * velocidad[10] + direction[10]), (0 * velocidad[11] + direction[11]), (1/(distance.euclidean(list(distancia[11]), list(distancia[12]))) * velocidad[11] + direction[11])],
       [(1/(distance.euclidean(list(distancia[12]), list(distancia[0]))) * velocidad[0] + direction[0]), (1/(distance.euclidean(list(distancia[12]), list(distancia[1]))) * velocidad[1] + direction[1]), (1/(distance.euclidean(list(distancia[12]), list(distancia[2]))) * velocidad[2] + direction[2]), (1/(distance.euclidean(list(distancia[12]), list(distancia[3]))) * velocidad[3] + direction[3]), (1/(distance.euclidean(list(distancia[12]), list(distancia[4]))) * velocidad[4] + direction[4]), (1/(distance.euclidean(list(distancia[12]), list(distancia[5]))) * velocidad[5] + direction[5]), (1/(distance.euclidean(list(distancia[12]), list(distancia[6]))) * velocidad[6] + direction[6]), (1/(distance.euclidean(list(distancia[0]), list(distancia[7]))) * velocidad[7] + direction[7]), (1/(distance.euclidean(list(distancia[12]), list(distancia[8]))) * velocidad[8] + direction[8]), (1/(distance.euclidean(list(distancia[12]), list(distancia[9]))) * velocidad[9] + direction[9]), (1/(distance.euclidean(list(distancia[12]), list(distancia[10]))) * velocidad[10] + direction[10]), (1/(distance.euclidean(list(distancia[12]), list(distancia[11]))) * velocidad[11] + direction[11]), (0 * velocidad[11] + direction[11])]])

b = (estimado_2017_11_01)
x = np.linalg.solve(A, b)


#######################################
#  Grafo Contaminacion
#######################################

G=nx.complete_graph(13)
pos=[(100.249,25.668000000000003),(100.255,25.75),(100.338,25.67),
     (100.366,25.756999999999998),(100.464,25.676),(100.586,25.783),
     (100.344,25.8),(100.18799999999999,25.776999999999997),(100.096,25.646),
     (100.413,25.665),(99.9955,25.36),(100.2489,25.5749),(100.3099,25.7295)]

weights1 = np.random.choice(estimado_2017_11_01, nx.number_of_edges(G))
w = 0
for u, v, d in G.edges(data=True):
    d['weight'] = weights1[w]
    w += 1
    
lweights1  = [math.log10(i) for i in weights1]
m = np.asarray(lweights1 )

colors = range(len(lweights1))
colors_nodes = range(len(G.nodes))

"""nx.draw(G, pos, node_color=colors_nodes, edge_color=colors,
        width=m, edge_cmap=plt.cm.Blues, cmap=plt.cm.Wistia,
        with_labels=False)
plt.savefig('grafo_cont.jpg', format='jpg', dpi=1000)

nx.draw(G, pos, node_color=colors_nodes, edge_color='silver',
        width=1, cmap=plt.cm.RdYlGn_r,
        with_labels=False)
plt.savefig('grafon.jpg', format='jpg', dpi=1000)"""

##########################################################
# wind rose
#########################################################
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from math import pi
from windrose import WindroseAxes

 

centro['velocidad_x'] = centro['velocity'] * np.sin(centro['direction'] * pi / 180.0)
centro['velocidad_y'] = centro['velocity'] * np.cos(centro['direction'] * pi / 180.0)
fig, ax = plt.subplots(figsize=(8, 8), dpi=80)
x0, x1 = ax.get_xlim()
y0, y1 = ax.get_ylim()
ax.set_aspect('equal')
_ = centro.plot(kind='scatter', x='velocidad_x', y='velocidad_y', alpha=0.35, ax=ax)


ax = WindroseAxes.from_ax()
ax.bar(centro.direction, centro.velocity, normed=True, opening=0.8, edgecolor='white')
ax.set_legend()
plt.title('Estación Centro')
plt.savefig('rose.jpg', format='jpg', dpi=1000)