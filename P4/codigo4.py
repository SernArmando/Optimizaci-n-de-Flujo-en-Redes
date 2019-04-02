import numpy as np
import pandas as pd
import time
import csv
from scipy.spatial import distance
import networkx as nx
import matplotlib.pyplot as plt
from time import time 
import statistics
import random
from datetime import datetime
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from statsmodels.formula.api import ols
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multicomp import MultiComparison


#########################################################
##################   GENERAR GRAFOS   ###################
#########################################################
aux=0
matriz=np.arange(1800*5,dtype=float).reshape(1800,5)
ordenes=[16,32,64,128]
for i in range((len(ordenes))*4):
    
    if i <=3:
        rango=random.randint(ordenes[i]*3,ordenes[i]*ordenes[i])
        G=nx.dense_gnm_random_graph(ordenes[i],rango)
        graf=1
    if 3<i<=7:
       rango=random.randint(1,9) 
       G=nx.gnp_random_graph(ordenes[i-4],(rango/10))
       graf=2
    if 7<i<=11:
        rango=random.randint(ordenes[i-8]*2,ordenes[i-8]*ordenes[i-8])
        G=nx.gnm_random_graph(ordenes[i-8],rango , seed=None, directed=False)   
        graf=3
    for j in range(10):
        lista=[]
        lista[:]=G.edges
        
        for r in range(len(lista)):
            R=np.random.normal(loc=20, scale=5.0, size=None)
            G.add_edge(lista[r][0],lista[r][1],capacity=R)
            
        for k in range(15):
            tiempo_inicial = time()
            initial=final=0
            while initial==final:
                initial=random.randint(0,round(len(G.nodes)/2))
                final=random.randint(initial,len(G.nodes)-2)
            if k<=4:
                T=nx.maximum_flow(G, initial, final)
                algorit=1
            if 4<k<=9:
                T=nx.algorithms.flow.edmonds_karp(G, initial, final)
                algorit=2
            if 9<k<=14:
                T=nx.algorithms.flow.boykov_kolmogorov(G,initial,final) 
                algorit=3
            tiempo_final = time()
            tiempo_ejecucion = tiempo_final - tiempo_inicial
            
            print(len(G.nodes))
            matriz[aux,0]=algorit
            matriz[aux,1]=graf
            matriz[aux,2]=len(G.nodes)
            matriz[aux,3]=nx.density(G)
            matriz[aux,4]=tiempo_ejecucion
            aux+=1

data=pd.DataFrame(matriz)
data.columns=['Algoritmo', 'Generador','Orden','Densidad','Tiempo']


#########################################################
##################         ANOVA      ###################
#########################################################

data.columns=['Algoritmo', 'Generador','Orden','Densidad','Tiempo'] 
model_name = ols('Tiempo ~ Algoritmo+Generador+Orden+Densidad', data=data).fit()
f = open('Ols.txt', "w")
f.write('%s \t' %model_name.summary())
f.close()

aov_table = sm.stats.anova_lm(model_name, typ=2)
f = open('Anova.txt', "w")
f.write('%s \t' %aov_table)
f.close()

#########################################################
##################    HISTOGRAMAS     ###################
#########################################################

"""data_filter= data[data['Generador'] == 1]
plt.hist(data_filter['Tiempo'], histtype='bar', color='blue')
plt.xlabel('Tiempo (segundos)', size=14)
plt.title('dense_gnm_random_graph',size=18, color='blue')
plt.grid(axis='y', color='silver')
plt.subplots_adjust(left=0.15)
plt.savefig('histograma1.eps', format='eps', dpi=1000)
    
 

data_filter= data[data['Generador'] == 2]
plt.hist(data_filter['Tiempo'], histtype='bar', color='pink')
plt.xlabel('Tiempo (segundos)', size=14)
plt.title('gnp_random_graph',size=18, color='pink')
plt.grid(axis='y', color='silver')
plt.subplots_adjust(left=0.15)
plt.savefig('histograma2.eps', format='eps', dpi=1000)



data_filter= data[data['Generador'] == 3]
plt.hist(data_filter['Tiempo'], histtype='bar', color='yellow')
plt.xlabel('Tiempo (segundos)', size=14)
plt.title('gnm_random_graph',size=18, color='yellow')
plt.grid(axis='y', color='silver')
plt.subplots_adjust(left=0.15)
plt.savefig('histograma3.eps', format='eps', dpi=1000)

#########################################################
##################    HISTOGRAMAS     ###################
#########################################################


data_filter= data[data['Algoritmo'] == 1]
plt.hist(data_filter['Tiempo'], histtype='bar', color='blue')
plt.xlabel('Tiempo (segundos)', size=14)
plt.title('maximum_flow',size=18, color='blue')
plt.grid(axis='y', color='silver')
plt.subplots_adjust(left=0.15)
plt.savefig('histograma4.eps', format='eps', dpi=1000)
    
 

data_filter= data[data['Algoritmo'] == 2]
plt.hist(data_filter['Tiempo'], histtype='bar', color='blue')
plt.xlabel('Tiempo (segundos)', size=14)
plt.title('algorithms.flow.edmonds_karp',size=18, color='blue')
plt.grid(axis='y', color='silver')
plt.subplots_adjust(left=0.15)
plt.savefig('histograma5.eps', format='eps', dpi=1000)



data_filter= data[data['Algoritmo'] == 3]
plt.hist(data_filter['Tiempo'], histtype='bar', color='blue')
plt.xlabel('Tiempo (segundos)', size=14)
plt.title('algorithms.flow.boycov_kolmogorov',size=18, color='blue')
plt.grid(axis='y', color='silver')
plt.subplots_adjust(left=0.15)
plt.savefig('histograma6.eps', format='eps', dpi=1000)


#########################################################
##################    HISTOGRAMAS     ###################
#########################################################


data_filter= data[data['Orden'] == 16]
plt.hist(data_filter['Tiempo'], histtype='bar', color='blue')
plt.xlabel('Tiempo (segundos)', size=14)
plt.title('orden (16)',size=18, color='blue')
plt.grid(axis='y', color='silver')
plt.subplots_adjust(left=0.15)
plt.savefig('histograma7.eps', format='eps', dpi=1000)
    
 

data_filter= data[data['Orden'] == 32]
plt.hist(data_filter['Tiempo'], histtype='bar', color='blue')
plt.xlabel('Tiempo (segundos)', size=14)
plt.title('orden (32)',size=18, color='blue')
plt.grid(axis='y', color='silver')
plt.subplots_adjust(left=0.15)
plt.savefig('histograma8.eps', format='eps', dpi=1000)



data_filter= data[data['Orden'] == 64]
plt.hist(data_filter['Tiempo'], histtype='bar', color='blue')
plt.xlabel('Tiempo (segundos)', size=14)
plt.title('orden (64)',size=18, color='blue')
plt.grid(axis='y', color='silver')
plt.subplots_adjust(left=0.15)
plt.savefig('histograma9.eps', format='eps', dpi=1000)


data_filter= data[data['Orden'] == 128]
plt.hist(data_filter['Tiempo'], histtype='bar', color='blue')
plt.xlabel('Tiempo (segundos)', size=14)
plt.title('orden (128)',size=18, color='blue')
plt.grid(axis='y', color='silver')
plt.subplots_adjust(left=0.15)
plt.savefig('histograma10.eps', format='eps', dpi=1000)"""

  

#########################################################
###############  MATRIZ DE CORRELACIONES   ##############
#########################################################
corr = data.corr()
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(corr,cmap='seismic', vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,len(data.columns),1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(( 'Alg.','Gen.', 'Ord.', 'Den.','Tie.'))
ax.set_yticklabels(( 'Algoritmo','Generador', 'Orden', 'Densidad','Tiempo'))
plt.title('Matriz de correlaciones', pad=16.0,size=10)
plt.savefig('amiga.eps', format='eps', dpi=1000)    




#########################################################
###################      SCATTER        #################
#########################################################
"""for color in ('blue', 'pink', 'yellow'):
    for marker in ('*','v','o'):
        if color=='blue':
            aux1=1
        elif color=='pink':
            aux1=2
        elif color=='yellow':
            aux1=3
        else:
            print('error')
        if marker=='d':
            aux2=1
        elif marker=='v':
            aux2=2
        elif marker=='o':
            aux2=3
        else:
            print('error')
        x=[i for i in range(len(matriz)) if (matriz[i,1]==aux1 and matriz[i,0]==aux2)]
        y=[matriz[i,4] for i in range(len(matriz)) if (matriz[i,1]==aux1 and matriz[i,0]==aux2)]
        plt.scatter(x, y, marker=marker, c=color)
        #plt.ylim(0,1)
        #plt.xlim(200,500)
plt.ylabel('Tiempo (segundos)', size=14)
plt.title('Grafo contra tiempo',size=18)
blue_patch = mpatches.Patch(color='blue', label='dense_gnm_random_graph')
green_patch = mpatches.Patch(color='pink', label='gnp_random_graph')
red_patch = mpatches.Patch(color='yellow', label='gnm_random_graph')
cuadrado_line = mlines.Line2D([], [],color='black', marker='*', markersize=10, label='maximum_flow')
triangulo_line = mlines.Line2D([], [],color='black', marker='v', markersize=10, label='algorithms.flow.edmonds_karp')
circulo_line = mlines.Line2D([], [],color='black', marker='o', markersize=10, label='algorithms.flow.boykov_kolmogorov')
plt.legend(handles=[blue_patch,green_patch,red_patch,cuadrado_line,triangulo_line,circulo_line],bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)        
plt.savefig('scatter.eps', format='eps', dpi=1000)"""





#########################################################
#############  DIAGRAMAS DE CAJA Y BIGOTE  ##############
#########################################################
data1= data[data['Generador'] == 1]
data2= data[data['Generador'] == 2]
data3= data[data['Generador'] == 3]

tiempos1= data1['Tiempo']
tiempos2= data2['Tiempo']
tiempos3= data3['Tiempo']


"""to_plot=[tiempos1, tiempos2, tiempos3]
fig=plt.figure(1,figsize=(9,6))
ax=fig.add_subplot(111)
bp=ax.boxplot(to_plot)
plt.xlabel('Generador de grafo', size=14)
plt.ylabel('Tiempo (segundos)', size=14)
plt.title('Generador contra tiempo',size=18)
plt.savefig('boxplotgenerador.eps', format='eps', dpi=1000)"""




data1= data[data['Algoritmo'] == 1]
data2= data[data['Algoritmo'] == 2]
data3= data[data['Algoritmo'] == 3]

tiempos1= data1['Tiempo']
tiempos2= data2['Tiempo']
tiempos3= data3['Tiempo']


"""to_plot=[tiempos1, tiempos2, tiempos3]
fig=plt.figure(1,figsize=(9,6))
ax=fig.add_subplot(111)
bp=ax.boxplot(to_plot)
plt.xlabel('Algoritmo de flujo máximo', size=14)
plt.ylabel('Tiempo (segundos)', size=14)
plt.title('Algoritmo contra tiempo',size=18)
plt.savefig('boxplotalgoritmo.eps', format='eps', dpi=1000)"""




data1= data[data['Orden'] == 16]
data2= data[data['Orden'] == 32]
data3= data[data['Orden'] == 64]
data4= data[data['Orden'] == 128]

tiempos1= data1['Tiempo']
tiempos2= data2['Tiempo']
tiempos3= data3['Tiempo']
tiempos4= data3['Tiempo']


"""to_plot=[tiempos1, tiempos2, tiempos3, tiempos4]
fig=plt.figure(1,figsize=(9,6))
ax=fig.add_subplot(111)
bp=ax.boxplot(to_plot)
plt.xlabel('Orden del grafo', size=14)
plt.ylabel('Tiempo (segundos)', size=14)
plt.title('Orden contra tiempo',size=18)
plt.savefig('boxplotorden.eps', format='eps', dpi=1000)"""






data1= data[data['Densidad'] <= 0.5]
data2= data[data['Densidad'] <= 0.6]
data3= data[data['Densidad'] <= 0.7]
data4= data[data['Densidad'] <= 0.8]
data5= data[data['Densidad'] <= 0.9]
data6= data[data['Densidad'] <= 1]

tiempos1= data1['Tiempo']
tiempos2= data2['Tiempo']
tiempos3= data3['Tiempo']
tiempos4= data4['Tiempo']
tiempos5= data5['Tiempo']
tiempos6= data6['Tiempo']


"""to_plot=[tiempos1, tiempos2, tiempos3, tiempos4, tiempos5, tiempos6]
fig=plt.figure(1,figsize=(9,6))
ax=fig.add_subplot(111)
bp=ax.boxplot(to_plot)
plt.xlabel('Densidad del grafo', size=14)
plt.ylabel('Tiempo (segundos)', size=14)
plt.title('Densidad contra tiempo',size=18)
plt.savefig('boxplotdensidad.eps', format='eps', dpi=1000)"""


