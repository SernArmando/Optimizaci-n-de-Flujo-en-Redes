import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from random import randint
import collections
import time




#############################################
#             GENERAR GRAFOS                #
#############################################
G1 = nx.random_geometric_graph(50,0.25)
pos1 = nx.get_node_attributes(G1, 'pos')

G2 = nx.random_geometric_graph(50,0.25)
pos2 = nx.get_node_attributes(G2, 'pos')

G3 = nx.random_geometric_graph(50,0.25)
pos3 = nx.get_node_attributes(G3, 'pos')

G4 = nx.random_geometric_graph(50,0.25)
pos4 = nx.get_node_attributes(G4, 'pos')

G5 = nx.random_geometric_graph(50,0.25)
pos5 = nx.get_node_attributes(G5, 'pos')




#############################################
#              DIBUJAR GRAFOS               #
#############################################

#        ASIGNAR PESOS A LAS ARISTAS        #
weights1 = np.random.normal(3,1, nx.number_of_edges(G1))
w = 0
for u, v, d in G1.edges(data=True):
    d['weight'] = weights1[w]
    w += 1
    
weights2 = np.random.normal(3,1, nx.number_of_edges(G2))
w = 0
for u, v, d in G2.edges(data=True):
    d['weight'] = weights2[w]
    w += 1
    
weights3 = np.random.normal(3,1, nx.number_of_edges(G3))
w = 0
for u, v, d in G3.edges(data=True):
    d['weight'] = weights3[w]
    w += 1
    
weights4 = np.random.normal(3,1, nx.number_of_edges(G4))
w = 0
for u, v, d in G4.edges(data=True):
    d['weight'] = weights4[w]
    w += 1
    
weights5 = np.random.normal(3,1, nx.number_of_edges(G5))
w = 0
for u, v, d in G5.edges(data=True):
    d['weight'] = weights5[w]
    w += 1

#     SELECCION DE FUENTES Y SUMIDEROS    #
f1 = {randint(0,49)}
s1 = {randint(0,49)}

f2 = {randint(0,49)}
s2 = {randint(0,49)}

f3 = {randint(0,49)}
s3 = {randint(0,49)}

f4 = {randint(0,49)}
s4 = {randint(0,49)}

f5 = {randint(0,49)}
s5 = {randint(0,49)}

#                 DIBUJOS                #
nx.draw(G1, node_color='blue', edge_color='silver',node_size=80, width=weights1,
        pos=pos1, with_labels=False, alpha= 0.7)
nx.draw_networkx_nodes(G1, pos1, nodelist=f1,node_size=150, node_color='green',
                       node_shape='d')
nx.draw_networkx_nodes(G1, pos1, nodelist=s1,node_size=150, node_color='red',
                       node_shape='d')
plt.savefig('grafo1.jpg', format='jpg', dpi=1000)


nx.draw(G2, node_color='blue', edge_color='silver',node_size=80, width=weights2,
        pos=pos2, with_labels=False, alpha= 0.7)
nx.draw_networkx_nodes(G2, pos2, nodelist=f2,node_size=150, node_color='green',
                       node_shape='d')
nx.draw_networkx_nodes(G2, pos2, nodelist=s2,node_size=150, node_color='red',
                       node_shape='d')
plt.savefig('grafo2.jpg', format='jpg', dpi=1000)


nx.draw(G3, node_color='blue', edge_color='silver',node_size=80, width=weights3,
        pos=pos3, with_labels=False, alpha= 0.7)
nx.draw_networkx_nodes(G3, pos3, nodelist=f3,node_size=150, node_color='green',
                       node_shape='d')
nx.draw_networkx_nodes(G3, pos3, nodelist=s3,node_size=150, node_color='red',
                       node_shape='d')
plt.savefig('grafo3.jpg', format='jpg', dpi=1000)


nx.draw(G4, node_color='blue', edge_color='silver',node_size=80, width=weights4,
        pos=pos4, with_labels=False, alpha= 0.7)
nx.draw_networkx_nodes(G4, pos4, nodelist=f4,node_size=150, node_color='green',
                       node_shape='d')
nx.draw_networkx_nodes(G4, pos4, nodelist=s4,node_size=150, node_color='red',
                       node_shape='d')
plt.savefig('grafo4.jpg', format='jpg', dpi=1000)


nx.draw(G5, node_color='blue', edge_color='silver',node_size=80, width=weights5,
        pos=pos5, with_labels=False, alpha= 0.7)
nx.draw_networkx_nodes(G5, pos5, nodelist=f5,node_size=150, node_color='green',
                       node_shape='d')
nx.draw_networkx_nodes(G5, pos5, nodelist=s5,node_size=150, node_color='red',
                       node_shape='d')
plt.savefig('grafo5.jpg', format='jpg', dpi=1000)



#############################################
#                ALGORITMOS                 #
#############################################

tiempos_grafo_1=[]
tiempos_grafo_2=[]
tiempos_grafo_3=[]
tiempos_grafo_4=[]
tiempos_grafo_5=[]

for i in range(0,30):
    inicio1 = time.time()
    degree_sequence1 = [d for n, d in G1.degree()]
    degreeCount1 = collections.Counter(degree_sequence1)
    deg1, cnt1 = zip(*degreeCount1.items())
    final1 = time.time()
    total1 = final1-inicio1
    tiempos_grafo_1.append(total1)

    inicio2 = time.time()
    degree_sequence2 = [d for n, d in G2.degree()]
    degreeCount2 = collections.Counter(degree_sequence2)
    deg2, cnt2 = zip(*degreeCount2.items())
    final2 = time.time()
    total2 = final2-inicio2
    tiempos_grafo_2.append(total2)

    inicio3 = time.time()
    degree_sequence3 = [d for n, d in G3.degree()]
    degreeCount3 = collections.Counter(degree_sequence3)
    deg3, cnt3 = zip(*degreeCount3.items())
    final3 = time.time()
    total3 = final3-inicio3
    tiempos_grafo_3.append(total3)

    inicio4 = time.time()
    degree_sequence4 = [d for n, d in G4.degree()]
    degreeCount4 = collections.Counter(degree_sequence4)
    deg4, cnt4 = zip(*degreeCount4.items())
    final4 = time.time()
    total4 = final4-inicio4
    tiempos_grafo_4.append(total4)

    inicio5 = time.time()
    degree_sequence5 = [d for n, d in G5.degree()]
    degreeCount5 = collections.Counter(degree_sequence5)
    deg5, cnt5 = zip(*degreeCount5.items())
    final5 = time.time()
    total5 = final5-inicio5
    tiempos_grafo_5.append(total5)

tiempos_grafo_1=np.asarray(tiempos_grafo_1)
tiempos_grafo_2=np.asarray(tiempos_grafo_2)
tiempos_grafo_3=np.asarray(tiempos_grafo_3)
tiempos_grafo_4=np.asarray(tiempos_grafo_4)
tiempos_grafo_5=np.asarray(tiempos_grafo_5)


to_plot=[tiempos_grafo_1,tiempos_grafo_2,tiempos_grafo_3,tiempos_grafo_4,tiempos_grafo_5]
fig=plt.figure(1,figsize=(9,6))
ax=fig.add_subplot(111)
bp=ax.boxplot(to_plot)
plt.xlabel('Grafo', size=14)
plt.ylabel('Tiempo (segundos)', size=14)
plt.title('Distribución de grado',size=18)
plt.savefig('boxplot1.eps', format='eps', dpi=1000)


#####################################################


tiempos_grafo_1=[]
tiempos_grafo_2=[]
tiempos_grafo_3=[]
tiempos_grafo_4=[]
tiempos_grafo_5=[]

for i in range(0,30):
    inicio1 = time.time()
    agrupamiento1 = nx.clustering(G1)
    agrupCount1 = collections.Counter(agrupamiento1)
    agru1, cnt1 = zip(*agrupCount1.items())
    final1 = time.time()
    total1 = final1-inicio1
    tiempos_grafo_1.append(total1)

    inicio2 = time.time()
    agrupamiento2 = nx.clustering(G2)
    agrupCount2 = collections.Counter(agrupamiento2)
    agru2, cnt2 = zip(*agrupCount2.items())
    final2 = time.time()
    total2 = final2-inicio2
    tiempos_grafo_2.append(total2)

    inicio3 = time.time()
    agrupamiento3 = nx.clustering(G3)
    agrupCount3 = collections.Counter(agrupamiento3)
    agru3, cnt3 = zip(*agrupCount3.items())
    final3 = time.time()
    total3 = final3-inicio3
    tiempos_grafo_3.append(total3)

    inicio4 = time.time()
    agrupamiento4 = nx.clustering(G4)
    agrupCount4 = collections.Counter(agrupamiento4)
    agru4, cnt4 = zip(*agrupCount4.items())
    final4 = time.time()
    total4 = final4-inicio4
    tiempos_grafo_4.append(total4)

    inicio5 = time.time()
    agrupamiento5 = nx.clustering(G5)
    agrupCount5 = collections.Counter(agrupamiento5)
    agru5, cnt5 = zip(*agrupCount5.items())
    final5 = time.time()
    total5 = final5-inicio5
    tiempos_grafo_5.append(total5)

tiempos_grafo_1=np.asarray(tiempos_grafo_1)
tiempos_grafo_2=np.asarray(tiempos_grafo_2)
tiempos_grafo_3=np.asarray(tiempos_grafo_3)
tiempos_grafo_4=np.asarray(tiempos_grafo_4)
tiempos_grafo_5=np.asarray(tiempos_grafo_5)


to_plot=[tiempos_grafo_1,tiempos_grafo_2,tiempos_grafo_3,tiempos_grafo_4,tiempos_grafo_5]
fig=plt.figure(1,figsize=(9,6))
ax=fig.add_subplot(111)
bp=ax.boxplot(to_plot)
plt.xlabel('Grafo', size=14)
plt.ylabel('Tiempo (segundos)', size=14)
plt.title('Coeficiente de agrupamiento',size=18)
plt.savefig('boxplot2.eps', format='eps', dpi=1000)

###################################################


tiempos_grafo_1=[]
tiempos_grafo_2=[]
tiempos_grafo_3=[]
tiempos_grafo_4=[]
tiempos_grafo_5=[]

for i in range(0,30):
    inicio1 = time.time()
    centralida_de_cercania1 = dict(nx.closeness_centrality(G1))
    dmin = 1
    ncenter = 0
    for n in pos1:
        x, y = pos1[n]
        d = (x - 0.5)**2 + (y - 0.5)**2
        if d < dmin:
            ncenter = n
            dmin = d
    p1 = dict(nx.single_source_shortest_path_length(G1, ncenter))
    final1 = time.time()
    total1 = final1-inicio1
    tiempos_grafo_1.append(total1)

    inicio2 = time.time()
    centralida_de_cercania2 = dict(nx.closeness_centrality(G2))
    dmin = 1
    ncenter = 0
    for n in pos2:
        x, y = pos2[n]
        d = (x - 0.5)**2 + (y - 0.5)**2
        if d < dmin:
            ncenter = n
            dmin = d
    p2 = dict(nx.single_source_shortest_path_length(G2, ncenter))
    final2 = time.time()
    total2 = final2-inicio2
    tiempos_grafo_2.append(total2)

    inicio3 = time.time()
    centralida_de_cercania3 = dict(nx.closeness_centrality(G3))
    dmin = 1
    ncenter = 0
    for n in pos3:
        x, y = pos3[n]
        d = (x - 0.5)**2 + (y - 0.5)**2
        if d < dmin:
            ncenter = n
            dmin = d
    p3 = dict(nx.single_source_shortest_path_length(G3, ncenter))
    final3 = time.time()
    total3 = final3-inicio3
    tiempos_grafo_3.append(total3)

    inicio4 = time.time()
    centralida_de_cercania4 = dict(nx.closeness_centrality(G4))
    dmin = 1
    ncenter = 0
    for n in pos4:
        x, y = pos4[n]
        d = (x - 0.5)**2 + (y - 0.5)**2
        if d < dmin:
            ncenter = n
            dmin = d
    p4 = dict(nx.single_source_shortest_path_length(G4, ncenter))
    final4 = time.time()
    total4 = final4-inicio4
    tiempos_grafo_4.append(total4)

    inicio5 = time.time()
    centralida_de_cercania5 = dict(nx.closeness_centrality(G5))
    dmin = 1
    ncenter = 0
    for n in pos5:
        x, y = pos5[n]
        d = (x - 0.5)**2 + (y - 0.5)**2
        if d < dmin:
            ncenter = n
            dmin = d
    p5 = dict(nx.single_source_shortest_path_length(G5, ncenter))
    final5 = time.time()
    total5 = final5-inicio5
    tiempos_grafo_5.append(total5)

tiempos_grafo_1=np.asarray(tiempos_grafo_1)
tiempos_grafo_2=np.asarray(tiempos_grafo_2)
tiempos_grafo_3=np.asarray(tiempos_grafo_3)
tiempos_grafo_4=np.asarray(tiempos_grafo_4)
tiempos_grafo_5=np.asarray(tiempos_grafo_5)


to_plot=[tiempos_grafo_1,tiempos_grafo_2,tiempos_grafo_3,tiempos_grafo_4,tiempos_grafo_5]
fig=plt.figure(1,figsize=(9,6))
ax=fig.add_subplot(111)
bp=ax.boxplot(to_plot)
plt.xlabel('Grafo', size=14)
plt.ylabel('Tiempo (segundos)', size=14)
plt.title('Centralidad de cercanía',size=18)
plt.savefig('boxplot3.eps', format='eps', dpi=1000)

###############################################


tiempos_grafo_1=[]
tiempos_grafo_2=[]
tiempos_grafo_3=[]
tiempos_grafo_4=[]
tiempos_grafo_5=[]

for i in range(0,30):
    inicio1 = time.time()
    centralida_de_carga1 = dict(nx.load_centrality(G1))
    dmin = 1
    ncenter = 0
    for n in pos1:
        x, y = pos1[n]
        d = (x - 0.5)**2 + (y - 0.5)**2
        if d < dmin:
            ncenter = n
            dmin = d
    p1 = dict(nx.single_source_shortest_path_length(G1, ncenter))
    final1 = time.time()
    total1 = final1-inicio1
    tiempos_grafo_1.append(total1)

    inicio2 = time.time()
    centralida_de_carga2 = dict(nx.load_centrality(G2))
    dmin = 1
    ncenter = 0
    for n in pos2:
        x, y = pos2[n]
        d = (x - 0.5)**2 + (y - 0.5)**2
        if d < dmin:
            ncenter = n
            dmin = d
    p2 = dict(nx.single_source_shortest_path_length(G2, ncenter))
    final2 = time.time()
    total2 = final2-inicio2
    tiempos_grafo_2.append(total2)

    inicio3 = time.time()
    centralida_de_carga3 = dict(nx.load_centrality(G3))
    dmin = 1
    ncenter = 0
    for n in pos3:
        x, y = pos3[n]
        d = (x - 0.5)**2 + (y - 0.5)**2
        if d < dmin:
            ncenter = n
            dmin = d
    p3 = dict(nx.single_source_shortest_path_length(G3, ncenter))
    final3 = time.time()
    total3 = final3-inicio3
    tiempos_grafo_3.append(total3)

    inicio4 = time.time()
    centralida_de_carga4 = dict(nx.load_centrality(G4))
    dmin = 1
    ncenter = 0
    for n in pos4:
        x, y = pos4[n]
        d = (x - 0.5)**2 + (y - 0.5)**2
        if d < dmin:
            ncenter = n
            dmin = d
    p4 = dict(nx.single_source_shortest_path_length(G4, ncenter))
    final4 = time.time()
    total4 = final4-inicio4
    tiempos_grafo_4.append(total4)

    inicio5 = time.time()
    centralida_de_carga5 = dict(nx.load_centrality(G5))
    dmin = 1
    ncenter = 0
    for n in pos5:
        x, y = pos5[n]
        d = (x - 0.5)**2 + (y - 0.5)**2
        if d < dmin:
            ncenter = n
            dmin = d
    p5 = dict(nx.single_source_shortest_path_length(G5, ncenter))
    final5 = time.time()
    total5 = final5-inicio5
    tiempos_grafo_5.append(total5)

tiempos_grafo_1=np.asarray(tiempos_grafo_1)
tiempos_grafo_2=np.asarray(tiempos_grafo_2)
tiempos_grafo_3=np.asarray(tiempos_grafo_3)
tiempos_grafo_4=np.asarray(tiempos_grafo_4)
tiempos_grafo_5=np.asarray(tiempos_grafo_5)


to_plot=[tiempos_grafo_1,tiempos_grafo_2,tiempos_grafo_3,tiempos_grafo_4,tiempos_grafo_5]
fig=plt.figure(1,figsize=(9,6))
ax=fig.add_subplot(111)
bp=ax.boxplot(to_plot)
plt.xlabel('Grafo', size=14)
plt.ylabel('Tiempo (segundos)', size=14)
plt.title('Centralidad de carga',size=18)
plt.savefig('boxplot4.eps', format='eps', dpi=1000)

        
########################################################      

        
tiempos_grafo_1=[]
tiempos_grafo_2=[]
tiempos_grafo_3=[]
tiempos_grafo_4=[]
tiempos_grafo_5=[]

for i in range(0,30):
    inicio1 = time.time()
    e1 = dict(nx.eccentricity(G1))
    final1 = time.time()
    total1 = final1-inicio1
    tiempos_grafo_1.append(total1)

    inicio2 = time.time()
    e2 = dict(nx.eccentricity(G2))
    final2 = time.time()
    total2 = final2-inicio2
    tiempos_grafo_2.append(total2)

    inicio3 = time.time()
    e3 = dict(nx.eccentricity(G3))
    final3 = time.time()
    total3 = final3-inicio3
    tiempos_grafo_3.append(total3)

    inicio4 = time.time()
    e4 = dict(nx.eccentricity(G4))
    final4 = time.time()
    total4 = final4-inicio4
    tiempos_grafo_4.append(total4)

    inicio5 = time.time()
    e5 = dict(nx.eccentricity(G5))
    final5 = time.time()
    total5 = final5-inicio5
    tiempos_grafo_5.append(total5)

tiempos_grafo_1=np.asarray(tiempos_grafo_1)
tiempos_grafo_2=np.asarray(tiempos_grafo_2)
tiempos_grafo_3=np.asarray(tiempos_grafo_3)
tiempos_grafo_4=np.asarray(tiempos_grafo_4)
tiempos_grafo_5=np.asarray(tiempos_grafo_5)


to_plot=[tiempos_grafo_1,tiempos_grafo_2,tiempos_grafo_3,tiempos_grafo_4,tiempos_grafo_5]
fig=plt.figure(1,figsize=(9,6))
ax=fig.add_subplot(111)
bp=ax.boxplot(to_plot)
plt.xlabel('Grafo', size=14)
plt.ylabel('Tiempo (segundos)', size=14)
plt.title('Exentricidad',size=18)
plt.savefig('boxplot5.eps', format='eps', dpi=1000)


###################################################### 

tiempos_grafo_1=[]
tiempos_grafo_2=[]
tiempos_grafo_3=[]
tiempos_grafo_4=[]
tiempos_grafo_5=[]

for i in range(0,30):
    inicio1 = time.time()
    pr1=nx.pagerank(G1,0.4)
    final1 = time.time()
    total1 = final1-inicio1
    tiempos_grafo_1.append(total1)

    inicio2 = time.time()
    pr2=nx.pagerank(G2,0.4)
    final2 = time.time()
    total2 = final2-inicio2
    tiempos_grafo_2.append(total2)

    inicio3 = time.time()
    pr3=nx.pagerank(G3,0.4)
    final3 = time.time()
    total3 = final3-inicio3
    tiempos_grafo_3.append(total3)

    inicio4 = time.time()
    pr4=nx.pagerank(G4,0.4)
    final4 = time.time()
    total4 = final4-inicio4
    tiempos_grafo_4.append(total4)

    inicio5 = time.time()
    pr5=nx.pagerank(G5,0.4)
    final5 = time.time()
    total5 = final5-inicio5
    tiempos_grafo_5.append(total5)

tiempos_grafo_1=np.asarray(tiempos_grafo_1)
tiempos_grafo_2=np.asarray(tiempos_grafo_2)
tiempos_grafo_3=np.asarray(tiempos_grafo_3)
tiempos_grafo_4=np.asarray(tiempos_grafo_4)
tiempos_grafo_5=np.asarray(tiempos_grafo_5)


to_plot=[tiempos_grafo_1,tiempos_grafo_2,tiempos_grafo_3,tiempos_grafo_4,tiempos_grafo_5]
fig=plt.figure(1,figsize=(9,6))
ax=fig.add_subplot(111)
bp=ax.boxplot(to_plot)
plt.xlabel('Grafo', size=14)
plt.ylabel('Tiempo (segundos)', size=14)
plt.title('Page Rank',size=18)
plt.savefig('boxplot6.eps', format='eps', dpi=1000)


#######################################################
mf1 = nx.maximum_flow(G1, 49, 26)

mf2 = nx.maximum_flow(G2, f2, s2)

mf3 = nx.maximum_flow(G3, f3, s3)

mf4 = nx.maximum_flow(G4, f4, s4)

mf5 = nx.maximum_flow(G5, f5, s5)

x = (1,2,3,4,5)
y = (2.48,4.19,5.31,3.53,3.03)

########################################################

fig, ax = plt.subplots()
plt.plot(x, y, 'o', color='b')
plt.ylabel("Máximo Flujo")
plt.xlabel("Grafo")
plt.savefig('mf.eps', format='eps', dpi=1000)



#                HSTOGRAMAS                 #
fig, ax = plt.subplots()
plt.bar(deg1, cnt1, width=0.80, color='b')
plt.title("Histograma")
plt.ylabel("Número de nodos")
plt.xlabel("Grado")
ax.set_xticks([d + 0.4 for d in deg1])
ax.set_xticklabels(deg1)
plt.axes([0.6, 0.6, 0.3, 0.3])
pos = nx.spring_layout(G1)
plt.axis('off')
nx.draw_networkx_nodes(G1, pos1, node_size=20)
nx.draw_networkx_edges(G1, pos1, alpha=0.4)
plt.savefig('a1g1.eps', format='eps', dpi=1000)

fig, ax = plt.subplots()
plt.bar(deg2, cnt2, width=0.80, color='b')
plt.title("Histograma")
plt.ylabel("Número de nodos")
plt.xlabel("Grado")
ax.set_xticks([d + 0.4 for d in deg2])
ax.set_xticklabels(deg2)
plt.axes([0.6, 0.6, 0.3, 0.3])
pos = nx.spring_layout(G2)
plt.axis('off')
nx.draw_networkx_nodes(G2, pos2, node_size=20)
nx.draw_networkx_edges(G2, pos2, alpha=0.4)
plt.show()
plt.savefig('a1g2.eps', format='eps', dpi=1000)


fig, ax = plt.subplots()
plt.bar(deg3, cnt3, width=0.80, color='b')
plt.title("Histograma")
plt.ylabel("Número de nodos")
plt.xlabel("Grado")
ax.set_xticks([d + 0.4 for d in deg3])
ax.set_xticklabels(deg3)
plt.axes([0.6, 0.6, 0.3, 0.3])
pos = nx.spring_layout(G3)
plt.axis('off')
nx.draw_networkx_nodes(G3, pos3, node_size=20)
nx.draw_networkx_edges(G3, pos3, alpha=0.4)
plt.show()
plt.savefig('a1g3.eps', format='eps', dpi=1000)


fig, ax = plt.subplots()
plt.bar(deg4, cnt4, width=0.80, color='b')
plt.title("Histograma")
plt.ylabel("Número de nodos")
plt.xlabel("Grado")
ax.set_xticks([d + 0.4 for d in deg4])
ax.set_xticklabels(deg4)
plt.axes([0.6, 0.6, 0.3, 0.3])
pos = nx.spring_layout(G4)
plt.axis('off')
nx.draw_networkx_nodes(G4, pos4, node_size=20)
nx.draw_networkx_edges(G4, pos4, alpha=0.4)
plt.show()
plt.savefig('a1g4.eps', format='eps', dpi=1000)


fig, ax = plt.subplots()
plt.bar(deg5, cnt5, width=0.80, color='b')
plt.title("Histograma")
plt.ylabel("Número de nodos")
plt.xlabel("Grado")
ax.set_xticks([d + 0.4 for d in deg5])
ax.set_xticklabels(deg5)
plt.axes([0.6, 0.6, 0.3, 0.3])
pos = nx.spring_layout(G5)
plt.axis('off')
nx.draw_networkx_nodes(G5, pos5, node_size=20)
nx.draw_networkx_edges(G5, pos5, alpha=0.4)
plt.show()
plt.savefig('a1g5.eps', format='eps', dpi=1000)




#                 CLUSTERING                #
fig, ax = plt.subplots()
plt.plot(agru1, cnt1, 'o', color='b')
plt.title("Coeficiente de agrupamiento")
plt.ylabel("Valor")
plt.xlabel("Nodo")
ax.set_xticks([d + 1 for d in agru1])
ax.set_xticklabels(agru1)
plt.savefig('a2g1.eps', format='eps', dpi=1000)


fig, ax = plt.subplots()
plt.plot(agru2, cnt2, 'o', color='b')
plt.title("Coeficiente de agrupamiento")
plt.ylabel("Valor")
plt.xlabel("Nodo")
ax.set_xticks([d + 1 for d in agru2])
ax.set_xticklabels(agru2)
plt.savefig('a2g2.eps', format='eps', dpi=1000)


fig, ax = plt.subplots()
plt.plot(agru3, cnt3, 'o', color='b')
plt.title("Coeficiente de agrupamiento")
plt.ylabel("Valor")
plt.xlabel("Nodo")
ax.set_xticks([d + 1 for d in agru3])
ax.set_xticklabels(agru3)
plt.savefig('a2g3.eps', format='eps', dpi=1000)


fig, ax = plt.subplots()
plt.plot(agru4, cnt4, 'o', color='b')
plt.title("Coeficiente de agrupamiento")
plt.ylabel("Valor")
plt.xlabel("Nodo")
ax.set_xticks([d + 1 for d in agru4])
ax.set_xticklabels(agru4)
plt.savefig('a2g4.eps', format='eps', dpi=1000)


fig, ax = plt.subplots()
plt.plot(agru5, cnt5, 'o', color='b')
plt.title("Coeficiente de agrupamiento")
plt.ylabel("Valor")
plt.xlabel("Nodo")
ax.set_xticks([d + 1 for d in agru5])
ax.set_xticklabels(agru5)
plt.savefig('a2g5.eps', format='eps', dpi=1000)







#  CENTRALIDAD DE CERCANIA Y DE CARGA      #
fig=plt.figure(figsize=(8, 8))
nx.draw_networkx_edges(G1, pos1, nodelist=[ncenter], alpha=1)
nx.draw_networkx_nodes(G1, pos1, nodelist=list(p1.keys()),
                       node_size=100,
                       node_color=list(p1.values()),
                       cmap=plt.cm.Blues_r)
fig.set_facecolor("plum")
plt.axis('off')
plt.savefig('a3g1.eps', format='eps', dpi=1000)


fig=plt.figure(figsize=(8, 8))
nx.draw_networkx_edges(G2, pos2, nodelist=[ncenter], alpha=1)
nx.draw_networkx_nodes(G2, pos2, nodelist=list(p2.keys()),
                       node_size=100,
                       node_color=list(p2.values()),
                       cmap=plt.cm.Blues_r)
fig.set_facecolor("plum")
plt.axis('off')
plt.savefig('a3g2.eps', format='eps', dpi=1000)


fig=plt.figure(figsize=(8, 8))
nx.draw_networkx_edges(G3, pos3, nodelist=[ncenter], alpha=1)
nx.draw_networkx_nodes(G3, pos3, nodelist=list(p3.keys()),
                       node_size=100,
                       node_color=list(p3.values()),
                       cmap=plt.cm.Blues_r)
fig.set_facecolor("plum")
plt.axis('off')
plt.savefig('a3g3.eps', format='eps', dpi=1000)


fig=plt.figure(figsize=(8, 8))
nx.draw_networkx_edges(G4, pos4, nodelist=[ncenter], alpha=1)
nx.draw_networkx_nodes(G4, pos4, nodelist=list(p4.keys()),
                       node_size=100,
                       node_color=list(p4.values()),
                       cmap=plt.cm.Blues_r)
fig.set_facecolor("plum")
plt.axis('off')
plt.savefig('a3g4.eps', format='eps', dpi=1000)


fig=plt.figure(figsize=(8, 8))
nx.draw_networkx_edges(G5, pos5, nodelist=[ncenter], alpha=1)
nx.draw_networkx_nodes(G5, pos5, nodelist=list(p5.keys()),
                       node_size=100,
                       node_color=list(p5.values()),
                       cmap=plt.cm.Blues_r)
fig.set_facecolor("plum")
plt.axis('off')
plt.savefig('a3g5.eps', format='eps', dpi=1000)






#               EXENTRICIDAD              #
fig=plt.figure(figsize=(8, 8))
nx.draw_networkx_edges(G1, pos1, nodelist=[ncenter], alpha=1)
nx.draw_networkx_nodes(G1, pos1, nodelist=list(p1.keys()),
                       node_size=100,
                       node_color=list(p1.values()),
                       cmap=plt.cm.Blues)
fig.set_facecolor("plum")
plt.axis('off')
plt.savefig('a5g1.eps', format='eps', dpi=1000)


fig=plt.figure(figsize=(8, 8))
nx.draw_networkx_edges(G2, pos2, nodelist=[ncenter], alpha=1)
nx.draw_networkx_nodes(G2, pos2, nodelist=list(p2.keys()),
                       node_size=100,
                       node_color=list(p2.values()),
                       cmap=plt.cm.Blues)
fig.set_facecolor("plum")
plt.axis('off')
plt.savefig('a5g2.eps', format='eps', dpi=1000)


fig=plt.figure(figsize=(8, 8))
nx.draw_networkx_edges(G3, pos3, nodelist=[ncenter], alpha=1)
nx.draw_networkx_nodes(G3, pos3, nodelist=list(p3.keys()),
                       node_size=100,
                       node_color=list(p3.values()),
                       cmap=plt.cm.Blues)
fig.set_facecolor("plum")
plt.axis('off')
plt.savefig('a5g3.eps', format='eps', dpi=1000)


fig=plt.figure(figsize=(8, 8))
nx.draw_networkx_edges(G4, pos4, nodelist=[ncenter], alpha=1)
nx.draw_networkx_nodes(G4, pos4, nodelist=list(p4.keys()),
                       node_size=100,
                       node_color=list(p4.values()),
                       cmap=plt.cm.Blues)
fig.set_facecolor("plum")
plt.axis('off')
plt.savefig('a5g4.eps', format='eps', dpi=1000)


fig=plt.figure(figsize=(8, 8))
nx.draw_networkx_edges(G5, pos5, nodelist=[ncenter], alpha=1)
nx.draw_networkx_nodes(G5, pos5, nodelist=list(p5.keys()),
                       node_size=100,
                       node_color=list(p5.values()),
                       cmap=plt.cm.Blues)
fig.set_facecolor("plum")
plt.axis('off')
plt.savefig('a5g5.eps', format='eps', dpi=1000)



