import networkx as nx
import matplotlib.pyplot as plt
from random import randint
import time
import numpy as np
import matplotlib.mlab as mlab
from scipy import stats
import pandas as pd

    
inicio_total = time.time() 
######    11.- Multigrafo dirigido aciclico      #####
######    Generar grafos     ######

G1 = nx.krackhardt_kite_graph(create_using=None)
G2 = nx.MultiDiGraph()
G2.add_edges_from([(1,1),(2,1),(3,1),
   (1,2),(4,2),(5,2),(6,2),
   (1,3),(2,3),(3,3),(4,3),(5,3),(6,3),
   (1,4),(2,4),(6,4),  
   (1,5),(2,5),(3,5),(4,5),(5,5),(6,5),
   (1,6),(5,6),(6,6)])

G3 = nx.balanced_tree(3, 4)
G4 = nx.generators.barabasi_albert_graph(100, 2)
G5 = nx.random_geometric_graph(50, 0.4)

######     Algoritmo betweenness_centrality   ######
tiempos_algoritmo_1=[]
tiempos_grafo_1=[]
tiempos_grafo_2=[]
tiempos_grafo_3=[]
tiempos_grafo_4=[]
tiempos_grafo_5=[]

for i in range(0,30):
    inicio_algoritmo_1 = time.time()
    for j in range(1,70):
        
        inicio1 = time.time()
        b_w_1 = nx.betweenness_centrality(G1)
        final1 = time.time()
        total1 = final1-inicio1
        tiempos_grafo_1.append(total1)
        
        inicio2 = time.time()
        b_w_2 = nx.betweenness_centrality(G2)
        final2 = time.time()
        total2 = final2-inicio2
        tiempos_grafo_2.append(total2)
        
        inicio3 = time.time()
        b_w_3 = nx.betweenness_centrality(G3)
        final3 = time.time()
        total3 = final3-inicio3
        tiempos_grafo_3.append(total3)
        
        inicio4 = time.time()
        b_w_4 = nx.betweenness_centrality(G4)
        final4 = time.time()
        total4 = final4-inicio4
        tiempos_grafo_4.append(total4)
        
        inicio5 = time.time()
        b_w_5 = nx.betweenness_centrality(G5)
        final5 = time.time()
        total5 = final5-inicio5
        tiempos_grafo_5.append(total5)
    final_algoritmo_1 = time.time()
    total_algoritmo_1 = (final_algoritmo_1 - inicio_algoritmo_1)
    tiempos_algoritmo_1.append(total_algoritmo_1)
    print(i, total_algoritmo_1)



######     Dibuja grafo     ######
"""nx.draw_networkx_nodes(G, pos,node_size=400,node_color='white')
nx.draw_networkx_edges(G, pos, width=2, edge_color='pink')
nx.draw_networkx_labels(G,pos,labels=b_w)
plt.axis('off')
plt.savefig('grafo11.eps', format='eps', dpi=1000)"""

"""nx.draw_networkx_nodes(H,pos, node_size=400,node_color='white')
nx.draw_networkx_edges(H, pos, width=2, edge_color='pink')
nx.draw_networkx_labels(H,pos, labels=be_we)
plt.axis('off')
plt.savefig('grafo12.eps', format='eps', dpi=1000)"""



tiempos_grafo_1=np.asarray(tiempos_grafo_1)
tiempos_grafo_2=np.asarray(tiempos_grafo_2)
tiempos_grafo_3=np.asarray(tiempos_grafo_3)
tiempos_grafo_4=np.asarray(tiempos_grafo_4)
tiempos_grafo_5=np.asarray(tiempos_grafo_5)



#####   Histograma por algoritmo    ######
tiempos_algoritmo_1=np.asarray(tiempos_algoritmo_1)

stats.shapiro(tiempos_algoritmo_1) #test shapiro-wilk


mu_1 = tiempos_algoritmo_1.mean()  # media de la distribución 
sigma_1 = tiempos_algoritmo_1.std()  # desviación estándar de la distribución 

"""n, bins, patches = plt.hist(tiempos_algoritmo_1, 'auto', density=True, facecolor='g', alpha=0.75)
y = mlab.normpdf(bins, mu_1, sigma_1)
plt.plot(bins, y, 'r--')
plt.xlabel('Tiempo (segundos)', size=14)
plt.ylabel('Frecuencia', size=14)
plt.title('Centralidad intermedia',size=18, color='green')
plt.text(5.12, 10, r'$N (\mu=5.11, \sigma = 0.04$)',color='r',size=14)
plt.grid(axis='y', color='silver')
plt.subplots_adjust(left=0.15)
plt.savefig('histograma1.eps', format='eps', dpi=1000)"""

######   8.- Multigrafo no dirigido ciclico    #####
######    Genera grafo     ######
"""G =nx.Graph()
pos = {0:(-1, 0), 1:(0,1), 2:(0,-1), 3:(1,0)}
G.add_edge(0,1,weight=2)
G.add_edge(0,2,weight=2)
G.add_edge(0,3,weight=1)
G.add_edge(1,3,weight=1)
G.add_edge(2,3,weight=1)"""


G6 = nx.generators.barabasi_albert_graph(100,5)
G7 = nx.random_geometric_graph(200, 0.3)
G8 = nx.krackhardt_kite_graph(create_using=None)
G9 = nx.generators.barabasi_albert_graph(50,3)
G10 = nx.Graph()
G10.add_edge(0,1,weight=2)
G10.add_edge(0,2,weight=2)
G10.add_edge(0,3,weight=1)
G10.add_edge(1,3,weight=1)
G10.add_edge(2,3,weight=1)


###### Algoritmo minimum_spanning_tree ######
"""T = nx.minimum_spanning_tree(G)"""
tiempos_algoritmo_2=[]
tiempos_grafo_6=[]
tiempos_grafo_7=[]
tiempos_grafo_8=[]
tiempos_grafo_9=[]
tiempos_grafo_10=[]

for i in range(0,30):
    inicio_algoritmo_2 = time.time()
    for j in range(1,500):
        
        inicio1 = time.time()
        T1 = nx.minimum_spanning_tree(G6)
        final1 = time.time()
        total1 = final1-inicio1
        tiempos_grafo_6.append(total1)
        
        inicio2 = time.time()
        T2 = nx.minimum_spanning_tree(G7)
        final2 = time.time()
        total2 = final2-inicio2
        tiempos_grafo_7.append(total2)
        
        inicio3 = time.time()
        T3 = nx.minimum_spanning_tree(G8)
        final3 = time.time()
        total3 = final3-inicio3
        tiempos_grafo_8.append(total3)
        
        inicio4 = time.time()
        T4 = nx.minimum_spanning_tree(G9)
        final4 = time.time()
        total4 = final4-inicio4
        tiempos_grafo_9.append(total4)
        
        inicio5 = time.time()
        T5 = nx.minimum_spanning_tree(G10)
        final5 = time.time()
        total5 = final5-inicio5
        tiempos_grafo_10.append(total5)
    final_algoritmo_2 = time.time()
    total_algoritmo_2 = (final_algoritmo_2 - inicio_algoritmo_2)
    tiempos_algoritmo_2.append(total_algoritmo_2)
    print(i, total_algoritmo_2)





######     Dibuja grafo     ######
    
    
tiempos_grafo_6=np.asarray(tiempos_grafo_6)
tiempos_grafo_7=np.asarray(tiempos_grafo_7)
tiempos_grafo_8=np.asarray(tiempos_grafo_8)
tiempos_grafo_9=np.asarray(tiempos_grafo_9)
tiempos_grafo_10=np.asarray(tiempos_grafo_10)



#####   Histograma por algoritmo    ######
tiempos_algoritmo_2=np.asarray(tiempos_algoritmo_2)

stats.shapiro(tiempos_algoritmo_2) #test shapiro-wilk


mu_2 = tiempos_algoritmo_2.mean()  # media de la distribución 
sigma_2 = tiempos_algoritmo_2.std()  # desviación estándar de la distribución 

"""n, bins, patches = plt.hist(tiempos_algoritmo_2, 'auto', density=True, facecolor='orange', alpha=0.75)
y = mlab.normpdf(bins, mu_2, sigma_2)
plt.plot(bins, y, 'r--')
plt.xlabel('Tiempo (segundos)', size=14)
plt.ylabel('Frecuencia', size=14)
plt.title('Árbol de expansión mínima',size=18, color='orange')
plt.text(6.2, 1, r'$N (\mu=5.96, \sigma = 0.20$)',color='r',size=14)
plt.grid(axis='y', color='silver')
plt.subplots_adjust(left=0.15)
plt.savefig('histograma2.eps', format='eps', dpi=1000)"""

"""nx.draw(B, node_size=40, node_color="blue", 
       pos= nx.circular_layout(B),width=1, edge_color='silver')
plt.savefig('grafo24.eps', format='eps', dpi=1000)

nx.draw(B, node_size=40, node_color="blue", 
       pos= nx.circular_layout(B),width=1, edge_color='silver')
nx.draw_networkx_nodes(T, nx.circular_layout(B),node_size=40,node_color='b')
nx.draw_networkx_edges(T, nx.circular_layout(B), width=2, edge_color='g')
plt.savefig('grafo25.eps', format='eps', dpi=1000)"""


"""nx.draw_networkx_nodes(B, pos,node_size=400,node_color='skyblue')
nx.draw_networkx_edges(B, pos, width=3, edge_color='silver')
nx.draw_networkx_labels(B,pos)
plt.axis('off')
plt.savefig('grafo22.eps', format='eps', dpi=1000)

nx.draw_networkx_nodes(T, pos,node_size=400,node_color='skyblue')
nx.draw_networkx_edges(T, pos, width=3, edge_color='silver')
nx.draw_networkx_labels(T,pos)
plt.axis('off')
plt.savefig('grafo23.eps', format='eps', dpi=1000)"""





######   10.- Multigrafo dirigido aciclico    #####
######    Genera grafo     ######
G11 = nx.DiGraph()
G12 = nx.DiGraph()
G13 = nx.DiGraph()
G14 = nx.DiGraph()
G15 = nx.DiGraph()

for i in range(1,50):
    for j in range(1,50):
        random = randint(1, 10)
        #print(i, j, random)
        if random == 1:
            G11.add_edge(i,j, capacidad=randint(1, 10))

for i in range(1,10):
    for j in range(1,10):
        random = randint(1, 10)
        #print(i, j, random)
        if random == 1:
            G12.add_edge(i,j, capacidad=randint(1, 10))

for i in range(1,100):
    for j in range(1,100):
        random = randint(1, 10)
        #print(i, j, random)
        if random == 1:
            G13.add_edge(i,j, capacidad=randint(1, 10))            

for i in range(1,150):
    for j in range(1,150):
        random = randint(1, 10)
        #print(i, j, random)
        if random == 1:
            G14.add_edge(i,j, capacidad=randint(1, 10))

for i in range(1,15):
    for j in range(1,15):
        random = randint(1, 10)
        #print(i, j, random)
        if random == 1:
            G15.add_edge(i,j, capacidad=randint(1, 10))    

######     Algoritmo  maximum_flow   ######  
tiempos_algoritmo_3=[]
tiempos_grafo_11=[]
tiempos_grafo_12=[]
tiempos_grafo_13=[]
tiempos_grafo_14=[]
tiempos_grafo_15=[]

for i in range(0,30):
    inicio_algoritmo_3 = time.time()
    for j in range(1,150):
        
        inicio1 = time.time()
        MF_1 = nx.maximum_flow(G11, 1, 10, capacity='capacidad', flow_func=None)
        final1 = time.time()
        total1 = final1-inicio1
        tiempos_grafo_11.append(total1)
        
        inicio2 = time.time()
        TMF_2 = nx.maximum_flow(G12, 1, 6, capacity='capacidad', flow_func=None)
        final2 = time.time()
        total2 = final2-inicio2
        tiempos_grafo_12.append(total2)
        
        inicio3 = time.time()
        MF_3 = nx.maximum_flow(G13, 1, 12, capacity='capacidad', flow_func=None)
        final3 = time.time()
        total3 = final3-inicio3
        tiempos_grafo_13.append(total3)
        
        inicio4 = time.time()
        MF_4 = nx.maximum_flow(G14, 1, 12, capacity='capacidad', flow_func=None)
        final4 = time.time()
        total4 = final4-inicio4
        tiempos_grafo_14.append(total4)
        
        inicio5 = time.time()
        MF_5 = nx.maximum_flow(G15, 1, 12, capacity='capacidad', flow_func=None)
        final5 = time.time()
        total5 = final5-inicio5
        tiempos_grafo_15.append(total5)
    final_algoritmo_3 = time.time()
    total_algoritmo_3 = (final_algoritmo_3 - inicio_algoritmo_3)
    tiempos_algoritmo_3.append(total_algoritmo_3)
    print(i, total_algoritmo_3)





######     Dibuja grafo     ######
    
    
tiempos_grafo_11=np.asarray(tiempos_grafo_11)
tiempos_grafo_12=np.asarray(tiempos_grafo_12)
tiempos_grafo_13=np.asarray(tiempos_grafo_13)
tiempos_grafo_14=np.asarray(tiempos_grafo_14)
tiempos_grafo_15=np.asarray(tiempos_grafo_15)



#####   Histograma por algoritmo    ######
tiempos_algoritmo_3=np.asarray(tiempos_algoritmo_3)

stats.shapiro(tiempos_algoritmo_3) #test shapiro-wilk


mu_3 = tiempos_algoritmo_3.mean()  # media de la distribución 
sigma_3 = tiempos_algoritmo_3.std()  # desviación estándar de la distribución 

"""n, bins, patches = plt.hist(tiempos_algoritmo_3, 'auto', density=True, facecolor='violet', alpha=0.75)
y = mlab.normpdf(bins, mu_3, sigma_3)
plt.plot(bins, y, 'r--')
plt.xlabel('Tiempo (segundos)', size=14)
plt.ylabel('Frecuencia', size=14)
plt.title('Flujo máximo',size=18, color='violet')
plt.text(5.2, 1, r'$N (\mu=5.25, \sigma = 0.55$)',color='r',size=14)
plt.grid(axis='y', color='silver')
plt.subplots_adjust(left=0.15)
plt.savefig('histograma3.eps', format='eps', dpi=1000)"""



######     Dibuja grafo     ######
"""nx.draw(G11, with_labels=True, node_size=400, node_color="aquamarine", pos=nx.circular_layout(G),
        width=1, edge_color='silver', font_size=12)
plt.axis('off')
plt.savefig('grafo31.eps', format='eps', dpi=1000)"""









######   ?.- Grafo no dirigido ciclico    #####
######    Genera grafo     ######
G16 = nx.Graph()
G17 = nx.Graph()
G18 = nx.Graph()
G19 = nx.Graph()
G20 = nx.Graph()


for i in range(1,15):
    for j in range(16,30):
                random = randint(1, 3)
                #print(i, j, random)
                if random ==1:
                    nx.add_path(G16, [i, j, randint(1, 9)])

for i in range(1,5):
    for j in range(6,10):
                random = randint(1, 3)
                #print(i, j, random)
                if random ==1:
                    nx.add_path(G17, [i, j, randint(10, 19)])

for i in range(1,25):
    for j in range(26,50):
                random = randint(1, 3)
                #print(i, j, random)
                if random ==1:
                    nx.add_path(G18, [i, j, randint(100, 200)])

for i in range(1,50):
    for j in range(51,100):
                random = randint(1, 3)
                #print(i, j, random)
                if random ==1:
                    nx.add_path(G19, [i, j, randint(200, 800)])

for i in range(1,100):
    for j in range(101,200):
                random = randint(1, 3)
                #print(i, j, random)
                if random ==1:
                    nx.add_path(G20, [i, j, randint(21, 29)])




######     Algoritmo   all_shortest_paths  #####
tiempos_algoritmo_4=[]
tiempos_grafo_16=[]
tiempos_grafo_17=[]
tiempos_grafo_18=[]
tiempos_grafo_19=[]
tiempos_grafo_20=[]

for i in range(0,30):
    inicio_algoritmo_4 = time.time()
    for j in range(1,2400):
        
        inicio1 = time.time()
        rutas1 = [p for p in nx.all_shortest_paths(G16, source=1, target=7, weight=None,method='dijkstra')]
        final1 = time.time()
        total1 = final1-inicio1
        tiempos_grafo_16.append(total1)
        
        inicio2 = time.time()
        rutas2 = [p for p in nx.all_shortest_paths(G17, source=1, target=7, weight=None,method='dijkstra')]
        final2 = time.time()
        total2 = final2-inicio2
        tiempos_grafo_17.append(total2)
        
        inicio3 = time.time()
        rutas3 = [p for p in nx.all_shortest_paths(G18, source=1, target=7, weight=None,method='dijkstra')]
        final3 = time.time()
        total3 = final3-inicio3
        tiempos_grafo_18.append(total3)
        
        inicio4 = time.time()
        rutas4 = [p for p in nx.all_shortest_paths(G19, source=1, target=7, weight=None,method='dijkstra')]
        final4 = time.time()
        total4 = final4-inicio4
        tiempos_grafo_19.append(total4)
        
        inicio5 = time.time()
        rutas5 = [p for p in nx.all_shortest_paths(G20, source=1, target=7, weight=None,method='dijkstra')]
        final5 = time.time()
        total5 = final5-inicio5
        tiempos_grafo_20.append(total5)
    final_algoritmo_4 = time.time()
    total_algoritmo_4 = (final_algoritmo_4 - inicio_algoritmo_4)
    tiempos_algoritmo_4.append(total_algoritmo_4)
    print(i, total_algoritmo_4)





######     Dibuja grafo     ######
    
    
tiempos_grafo_16=np.asarray(tiempos_grafo_16)
tiempos_grafo_17=np.asarray(tiempos_grafo_17)
tiempos_grafo_18=np.asarray(tiempos_grafo_18)
tiempos_grafo_19=np.asarray(tiempos_grafo_19)
tiempos_grafo_20=np.asarray(tiempos_grafo_20)



#####   Histograma por algoritmo    ######
tiempos_algoritmo_4=np.asarray(tiempos_algoritmo_4)

stats.shapiro(tiempos_algoritmo_4) #test shapiro-wilk


mu_4 = tiempos_algoritmo_4.mean()  # media de la distribución 
sigma_4 = tiempos_algoritmo_4.std()  # desviación estándar de la distribución 

"""n, bins, patches = plt.hist(tiempos_algoritmo_4, 'auto', density=True, facecolor='aqua', alpha=0.75)
y = mlab.normpdf(bins, mu_4, sigma_4)
plt.plot(bins, y, 'r--')
plt.xlabel('Tiempo (segundos)', size=14)
plt.ylabel('Frecuencia', size=14)
plt.title('Ruta mas corta',size=18, color='aqua')
plt.text(5.7, 14, r'$N (\mu=5.70, \sigma = 0.03$)',color='r',size=14)
plt.grid(axis='y', color='silver')
plt.subplots_adjust(left=0.15)
plt.savefig('histograma4.eps', format='eps', dpi=1000)"""




######     Dibuja grafo     ######
"""pos=nx.kamada_kawai_layout(Y)

nx.draw(Y, pos, node_color='palegreen', node_size=200,edge_color='silver')
nx.draw_networkx_labels(Y, pos, font_size=12, font_color='black')
plt.savefig('grafo41.eps', format='eps', dpi=1000)


nx.draw(Y, pos, node_color='palegreen', node_size=200,edge_color='silver')
nx.draw_networkx_edges(Y, pos, edgelist=[(1,15), (15,4 ), (4,16), (16, 7)],
                                         width=3, edge_color='blue')
nx.draw_networkx_edges(Y, pos, edgelist=[(1,11), (11,23), (23,20), (20, 7)],
                                         width=4, edge_color='red')
nx.draw_networkx_edges(Y, pos, edgelist=[(1,14), (14,5 ), (5,20), (20, 7)],
                                         width=2, edge_color='yellow')
nx.draw_networkx_labels(Y, pos, font_size=12, font_color='black')
plt.savefig('grafo42.eps', format='eps', dpi=1000)"""









######   ?.- Grafo no dirigido ciclico    #####
######    Genera grafo     ######
"""G=nx.Graph()
G.add_nodes_from(["Bacteria","a", "b", "c","d","e", "f", "g", "h"])
G.add_edges_from([("Bacteria","a"), ("Bacteria","b"),("Bacteria","c"),("Bacteria","d"),
                  ("Bacteria","e"),("Bacteria","f"),("Bacteria","g"),("Bacteria","h")])"""

G21 = nx.balanced_tree(3,5)
G22 = nx.krackhardt_kite_graph(create_using=None)
G23 = nx.MultiDiGraph()
G23.add_edges_from([(1,1),(2,1),(3,1),
   (1,2),(4,2),(5,2),(6,2),
   (1,3),(2,3),(3,3),(4,3),(5,3),(6,3),
   (1,4),(2,4),(6,4),  
   (1,5),(2,5),(3,5),(4,5),(5,5),(6,5),
   (1,6),(5,6),(6,6)])


G24 = nx.generators.barabasi_albert_graph(100, 3)
G25 = nx.random_geometric_graph(150, 0.3)

"""pos=nx.kamada_kawai_layout(G)
nx.draw(G,pos,node_size=20,alpha=0.5,node_color="blue", with_labels=False)
plt.axis('equal')"""

    
"""pos = nx.circular_layout(G)"""
    
"""d = nx.coloring.greedy_color(G, strategy='largest_first')"""



######     Algoritmo   greedy_color  #####
tiempos_algoritmo_5=[]
tiempos_grafo_21=[]
tiempos_grafo_22=[]
tiempos_grafo_23=[]
tiempos_grafo_24=[]
tiempos_grafo_25=[]

for i in range(0,30):
    inicio_algoritmo_5 = time.time()
    for j in range(1,3000):
        
        inicio1 = time.time()
        greedy1 = nx.coloring.greedy_color(G21, strategy='random_sequential')
        final1 = time.time()
        total1 = final1-inicio1
        tiempos_grafo_21.append(total1)
        
        inicio2 = time.time()
        greedy2 = nx.coloring.greedy_color(G22, strategy='random_sequential')
        final2 = time.time()
        total2 = final2-inicio2
        tiempos_grafo_22.append(total2)
        
        inicio3 = time.time()
        greedy3 = nx.coloring.greedy_color(G23, strategy='random_sequential')
        final3 = time.time()
        total3 = final3-inicio3
        tiempos_grafo_23.append(total3)
        
        inicio4 = time.time()
        greedy4 = nx.coloring.greedy_color(G24, strategy='random_sequential')
        final4 = time.time()
        total4 = final4-inicio4
        tiempos_grafo_24.append(total4)
        
        inicio5 = time.time()
        greedy5 = nx.coloring.greedy_color(G25, strategy='random_sequential')
        final5 = time.time()
        total5 = final5-inicio5
        tiempos_grafo_25.append(total5)
    final_algoritmo_5 = time.time()
    total_algoritmo_5 = (final_algoritmo_5 - inicio_algoritmo_5)
    tiempos_algoritmo_5.append(total_algoritmo_5)
    print(i, total_algoritmo_5)





######     Dibuja grafo     ######
    
    
tiempos_grafo_21=np.asarray(tiempos_grafo_21)
tiempos_grafo_22=np.asarray(tiempos_grafo_22)
tiempos_grafo_23=np.asarray(tiempos_grafo_23)
tiempos_grafo_24=np.asarray(tiempos_grafo_24)
tiempos_grafo_25=np.asarray(tiempos_grafo_25)



#####   Histograma por algoritmo    ######
tiempos_algoritmo_5=np.asarray(tiempos_algoritmo_5)

stats.shapiro(tiempos_algoritmo_5) #test shapiro-wilk


mu_5 = tiempos_algoritmo_5.mean()  # media de la distribución 
sigma_5 = tiempos_algoritmo_5.std()  # desviación estándar de la distribución 

"""n, bins, patches = plt.hist(tiempos_algoritmo_5, 'auto', density=True, facecolor='gray', alpha=0.75)
y = mlab.normpdf(bins, mu_5, sigma_5)
plt.plot(bins, y, 'r--')
plt.xlabel('Tiempo (segundos)', size=14)
plt.ylabel('Frecuencia', size=14)
plt.title('Coloración glotona',size=18, color='gray')
plt.text(5.18, 17, r'$N (\mu=5.18, \sigma = 0.02$)',color='r',size=14)
plt.grid(axis='y', color='silver')
plt.subplots_adjust(left=0.15)
plt.savefig('histograma5.eps', format='eps', dpi=1000)"""


######     Dibuja grafo     ######
"""node1=set()
node2=set()
node3=set()
node4=set()"""


        

"""for tipo, valores in d.items():
    if valores == 0:
        aux1=tipo
        node1.add(tipo)
    elif valores == 1:
        aux2=tipo
        node2.add(tipo)
    elif valores == 2:
        aux3=tipo
        node3.add(tipo)
    else:
         aux4=tipo
         node4.add(tipo)"""

"""lista1=[]
lista2=[]



for tipo, valores in d.items():
   lista1.append(tipo)
   lista2.append(valores)"""

"""nx.draw_networkx_nodes(G, pos, nodelist=node1, node_size=3000, node_color='skyblue')
nx.draw_networkx_nodes(G, pos, nodelist=node2, node_size=400, node_color='pink')
nx.draw_networkx_edges(G, pos, width=3, edge_color='silver')
nx.draw_networkx_labels(G, pos, font_size=18)"""



"""nx.draw_networkx_nodes(G, pos, nodelist=node1, node_size=5, node_color='blue')
nx.draw_networkx_nodes(G, pos, nodelist=node2, node_size=20, node_color='lime')
nx.draw_networkx_nodes(G, pos, nodelist=node3, node_size=60, node_color='magenta')
nx.draw_networkx_nodes(G, pos, nodelist=node4, node_size=90, node_color='red')
nx.draw_networkx_edges(G, pos, width=1, edge_color='silver')
plt.axis('off')
plt.savefig('grafo52.eps', format='eps', dpi=1000)"""

final_total = time.time()
total_total = (final_total-inicio_total)/60
print(total_total)





#####     Generar scateerplot     #####

##### Numero de nodos en los grafos  #####
N1 = len(G1.nodes)
N2 = len(G2.nodes)
N3= len(G3.nodes)
N4 = len(G4.nodes)
N5 = len(G5.nodes)
N6 = len(G6.nodes)
N7 = len(G7.nodes)
N8 = len(G8.nodes)
N9 = len(G9.nodes)
N10 = len(G10.nodes)
N11 = len(G11.nodes)
N12 = len(G12.nodes)
N13 = len(G13.nodes)
N14 = len(G14.nodes)
N15 = len(G15.nodes)
N16 = len(G16.nodes)
N17 = len(G17.nodes)
N18 = len(G18.nodes)
N19 = len(G19.nodes)
N20 = len(G20.nodes)
N21 = len(G21.nodes)
N22 = len(G22.nodes)
N23 = len(G23.nodes)
N24 = len(G24.nodes)
N25 = len(G25.nodes)



##### Numero de aristas en los grafos #####
V1 = len(G1.edges)
V2 = len(G2.edges)
V3 = len(G3.edges)
V4 = len(G4.edges)
V5 = len(G5.edges)
V6 = len(G6.edges)
V7 = len(G7.edges)
V8 = len(G8.edges)
V9 = len(G9.edges)
V10 = len(G10.edges)
V11 = len(G11.edges)
V12 = len(G12.edges)
V13 = len(G13.edges)
V14 = len(G14.edges)
V15 = len(G15.edges)
V16 = len(G16.edges)
V17 = len(G17.edges)
V18 = len(G18.edges)
V19 = len(G19.edges)
V20 = len(G20.edges)
V21 = len(G21.edges)
V22 = len(G22.edges)
V23 = len(G23.edges)
V24 = len(G24.edges)
V25 = len(G25.edges)

##### Medias de tiempo por algoritmo #####
mu_1_1 = tiempos_grafo_1.mean()
mu_1_2 = tiempos_grafo_2.mean()
mu_1_3 = tiempos_grafo_3.mean()
mu_1_4 = tiempos_grafo_4.mean()
mu_1_5 = tiempos_grafo_5.mean()
mu_2_1 = tiempos_grafo_6.mean()
mu_2_2 = tiempos_grafo_7.mean()
mu_2_3 = tiempos_grafo_8.mean()
mu_2_4 = tiempos_grafo_9.mean()
mu_2_5 = tiempos_grafo_10.mean()
mu_3_1 = tiempos_grafo_11.mean()
mu_3_2 = tiempos_grafo_12.mean()
mu_3_3 = tiempos_grafo_13.mean()
mu_3_4 = tiempos_grafo_14.mean()
mu_3_5 = tiempos_grafo_15.mean()
mu_4_1 = tiempos_grafo_16.mean()
mu_4_2 = tiempos_grafo_17.mean()
mu_4_3 = tiempos_grafo_18.mean()
mu_4_4 = tiempos_grafo_19.mean()
mu_4_5 = tiempos_grafo_20.mean()
mu_5_1 = tiempos_grafo_21.mean()
mu_5_2 = tiempos_grafo_22.mean()
mu_5_3 = tiempos_grafo_23.mean()
mu_5_4 = tiempos_grafo_24.mean()
mu_5_5 = tiempos_grafo_25.mean()



##### desv estan de tiempo por algoritmo #####
sigma_1_1 = tiempos_grafo_1.std()
sigma_1_2 = tiempos_grafo_2.std()
sigma_1_3 = tiempos_grafo_3.std()
sigma_1_4 = tiempos_grafo_4.std()
sigma_1_5 = tiempos_grafo_5.std()
sigma_2_1 = tiempos_grafo_6.std()
sigma_2_2 = tiempos_grafo_7.std()
sigma_2_3 = tiempos_grafo_8.std()
sigma_2_4 = tiempos_grafo_9.std()
sigma_2_5 = tiempos_grafo_10.std()
sigma_3_1 = tiempos_grafo_11.std()
sigma_3_2 = tiempos_grafo_12.std()
sigma_3_3 = tiempos_grafo_13.std()
sigma_3_4 = tiempos_grafo_14.std()
sigma_3_5 = tiempos_grafo_15.std()
sigma_4_1 = tiempos_grafo_16.std()
sigma_4_2 = tiempos_grafo_17.std()
sigma_4_3 = tiempos_grafo_18.std()
sigma_4_4 = tiempos_grafo_19.std()
sigma_4_5 = tiempos_grafo_20.std()
sigma_5_1 = tiempos_grafo_21.std()
sigma_5_2 = tiempos_grafo_22.std()
sigma_5_3 = tiempos_grafo_23.std()
sigma_5_4 = tiempos_grafo_24.std()
sigma_5_5 = tiempos_grafo_25.std()


##### nodos versus tiempos #####
##### nodos #####
x_1_1 = ([N1,N2,N3,N4,N5])
x_1_2 = ([N6,N7,N8,N9,N10])
x_1_3 = ([N11,N12,N13,N14,N15])
x_1_4 = ([N16,N17,N18,N19,N20])
x_1_5 = ([N21,N22,N23,N24,N25])

##### medias #####
y_1 = ([mu_1_1,mu_1_2,mu_1_3,mu_1_4,mu_1_5])
y_2 = ([mu_2_1,mu_2_2,mu_2_3,mu_3_4,mu_2_5])
y_3 = ([mu_3_1,mu_3_2,mu_3_3,mu_3_4,mu_3_5])
y_4 = ([mu_4_1,mu_4_2,mu_4_3,mu_4_4,mu_4_5])
y_5 = ([mu_5_1,mu_5_2,mu_5_3,mu_5_4,mu_5_5])

##### desv est #####
s_1 = ([sigma_1_1,sigma_1_2,sigma_1_3,sigma_1_4,sigma_1_5])
s_2 = ([sigma_2_1,sigma_2_2,sigma_2_3,sigma_3_4,sigma_2_5])
s_3 = ([sigma_3_1,sigma_3_2,sigma_3_3,sigma_3_4,sigma_3_5])
s_4 = ([sigma_4_1,sigma_4_2,sigma_4_3,sigma_4_4,sigma_4_5])
s_5 = ([sigma_5_1,sigma_5_2,sigma_5_3,sigma_5_4,sigma_5_5])



"""plt.errorbar(y_1,x_1_1, xerr=s_1, fmt='+',color='g',alpha=0.5)
plt.errorbar(y_2,x_1_2, xerr=s_2, fmt='o',color='orange',alpha=0.5)
plt.errorbar(y_3,x_1_3, xerr=s_3, fmt='^',color='violet',alpha=0.5)
plt.errorbar(y_4,x_1_4, xerr=s_4, fmt='>',color='aqua',alpha=0.5)
plt.errorbar(y_5,x_1_5, xerr=s_5, fmt='<',color='gray',alpha=0.5)
plt.xlabel('Tiempo (segundos)', size=14)
plt.ylabel('Cantidad de nodos', size=14)
plt.title('Cantidad de nodos contra tiempo',size=18)
plt.text(0.04, 200, r'Centralidad intermedia',color='g',size=14)
plt.text(0.04, 150, r'Árbol de expansión mínimo',color='orange',size=14)
plt.text(0.04, 100, r'Flujo máximo',color='violet',size=14)
plt.text(0.04, 50, r'Ruta mas corta',color='aqua',size=14)
plt.text(0.04, 0, r'Coloración  glotona',color='gray',size=14)
plt.text(0.04, 300, r'Algoritmo',color='black',size=18)
plt.savefig('scater1.eps', format='eps', dpi=1000)"""

###### zoom #####
"""plt.errorbar(y_1,x_1_1, xerr=s_1, fmt='+',color='g',alpha=0.5)
plt.errorbar(y_2,x_1_2, xerr=s_2, fmt='o',color='orange',alpha=0.5)
plt.errorbar(y_3,x_1_3, xerr=s_3, fmt='^',color='violet',alpha=0.5)
plt.errorbar(y_4,x_1_4, xerr=s_4, fmt='>',color='aqua',alpha=0.5)
plt.errorbar(y_5,x_1_5, xerr=s_5, fmt='<',color='gray',alpha=0.5)
plt.xlim(0.0,0.05)
plt.savefig('scater12.eps', format='eps', dpi=1000)"""



##### aristas #####
x_2_1 = ([V1,V2,V3,V4,V5])
x_2_2 = ([V6,V7,V8,V9,V10])
x_2_3 = ([V11,V12,V13,V14,V15])
x_2_4 = ([V16,V17,V18,V19,V20])
x_2_5 = ([V21,V22,V23,V24,V25])

"""plt.errorbar(y_1,x_2_1, xerr=s_1, fmt='+',color='g',alpha=0.5)
plt.errorbar(y_2,x_2_2, xerr=s_2, fmt='o',color='orange',alpha=0.5)
plt.errorbar(y_3,x_2_3, xerr=s_3, fmt='^',color='violet',alpha=0.5)
plt.errorbar(y_4,x_2_4, xerr=s_4, fmt='>',color='aqua',alpha=0.5)
plt.errorbar(y_5,x_2_5, xerr=s_5, fmt='<',color='gray',alpha=0.5)
plt.xlabel('Tiempo (segundos)', size=14)
plt.ylabel('Cantidad de arcos', size=14)
plt.title('Cantidad de arcos contra tiempo',size=18)
plt.text(0.04, 2000, r'Centralidad intermedia',color='g',size=14)
plt.text(0.04, 1500, r'Árbol de expansión mínimo',color='orange',size=14)
plt.text(0.04, 1000, r'Flujo máximo',color='violet',size=14)
plt.text(0.04, 500, r'Ruta mas corta',color='aqua',size=14)
plt.text(0.04, 0, r'Coloración  glotona',color='gray',size=14)
plt.text(0.04, 3000, r'Algoritmo',color='black',size=18)
plt.savefig('scater2.eps', format='eps', dpi=1000)"""


##### zoom #####
"""plt.errorbar(y_1,x_2_1, xerr=s_1, fmt='+',color='g',alpha=0.5)
plt.errorbar(y_2,x_2_2, xerr=s_2, fmt='o',color='orange',alpha=0.5)
plt.errorbar(y_3,x_2_3, xerr=s_3, fmt='^',color='violet',alpha=0.5)
plt.errorbar(y_4,x_2_4, xerr=s_4, fmt='>',color='aqua',alpha=0.5)
plt.errorbar(y_5,x_2_5, xerr=s_5, fmt='<',color='gray',alpha=0.5)
plt.xlim(0.0,0.05)
plt.savefig('scater22.eps', format='eps', dpi=1000)"""


###### boxplot  #####
"""to_plot=[tiempos_grafo_2,tiempos_grafo_1,tiempos_grafo_5,tiempos_grafo_4,tiempos_grafo_3]
fig=plt.figure(1,figsize=(9,6))
ax=fig.add_subplot(111)
bp=ax.boxplot(to_plot)
plt.xlabel('Grafo', size=14)
plt.ylabel('Tiempo (segundos)', size=14)
plt.title('Centralidad intermedia',size=18)
plt.text(0.9,0.003 , len(G2.nodes),color='r',size=14)
plt.text(1.9,0.003 , len(G1.nodes),color='r',size=14)
plt.text(2.9,0.015 , len(G5.nodes),color='r',size=14)
plt.text(3.9,0.032 , len(G4.nodes),color='r',size=14)
plt.text(5.1,0.039 , len(G3.nodes),color='r',size=14)
plt.savefig('boxplot1.eps', format='eps', dpi=1000)



to_plot=[tiempos_grafo_10,tiempos_grafo_8,tiempos_grafo_9,tiempos_grafo_6,tiempos_grafo_7]
fig=plt.figure(1,figsize=(9,6))
ax=fig.add_subplot(111)
bp=ax.boxplot(to_plot)
plt.xlabel('Grafo', size=14)
plt.ylabel('Tiempo (segundos)', size=14)
plt.title('Árbol de expansión mínima',size=18)
plt.text(0.9,0.005 , len(G10.nodes),color='r',size=14)
plt.text(1.9,0.005 , len(G8.nodes),color='r',size=14)
plt.text(2.9,0.005 , len(G9.nodes),color='r',size=14)
plt.text(3.9,0.01 , len(G6.nodes),color='r',size=14)
plt.text(4.9,0.015 , len(G7.nodes),color='r',size=14)
plt.savefig('boxplot2.eps', format='eps', dpi=1000)



to_plot=[tiempos_grafo_12,tiempos_grafo_15,tiempos_grafo_11,tiempos_grafo_13,tiempos_grafo_14]
fig=plt.figure(1,figsize=(9,6))
ax=fig.add_subplot(111)
bp=ax.boxplot(to_plot)
plt.xlabel('Grafo', size=14)
plt.ylabel('Tiempo (segundos)', size=14)
plt.title('Flujo máximo',size=18)
plt.text(0.9,0.005 , len(G12.nodes),color='r',size=14)
plt.text(1.9,0.005 , len(G15.nodes),color='r',size=14)
plt.text(2.9,0.01 , len(G11.nodes),color='r',size=14)
plt.text(3.9,0.015 , len(G13.nodes),color='r',size=14)
plt.text(4.9,0.02 , len(G14.nodes),color='r',size=14)
plt.savefig('boxplot3.eps', format='eps', dpi=1000)



to_plot=[tiempos_grafo_17,tiempos_grafo_16,tiempos_grafo_18,tiempos_grafo_20,tiempos_grafo_19]
fig=plt.figure(1,figsize=(9,6))
ax=fig.add_subplot(111)
bp=ax.boxplot(to_plot)
plt.xlabel('Grafo', size=14)
plt.ylabel('Tiempo (segundos)', size=14)
plt.title('Ruta mas corta',size=18)
plt.text(0.9,0.0013 , len(G17.nodes),color='r',size=14)
plt.text(1.9,0.0013 , len(G16.nodes),color='r',size=14)
plt.text(2.9,0.0013 , len(G18.nodes),color='r',size=14)
plt.text(3.9,0.0023 , len(G20.nodes),color='r',size=14)
plt.text(4.9,0.0023 , len(G19.nodes),color='r',size=14)
plt.savefig('boxplot4.eps', format='eps', dpi=1000)




to_plot=[tiempos_grafo_23,tiempos_grafo_22,tiempos_grafo_24,tiempos_grafo_25,tiempos_grafo_21]
fig=plt.figure(1,figsize=(9,6))
ax=fig.add_subplot(111)
bp=ax.boxplot(to_plot)
plt.xlabel('Grafo', size=14)
plt.ylabel('Tiempo (segundos)', size=14)
plt.title('Coloración glotona',size=18)
plt.text(0.9,0.00125 , len(G23.nodes),color='r',size=14)
plt.text(1.9,0.00125 , len(G22.nodes),color='r',size=14)
plt.text(2.9,0.00175 , len(G24.nodes),color='r',size=14)
plt.text(3.9,0.00175 , len(G25.nodes),color='r',size=14)
plt.text(4.9,0.00175, len(G21.nodes),color='r',size=14)
plt.savefig('boxplot5.eps', format='eps', dpi=1000)"""















