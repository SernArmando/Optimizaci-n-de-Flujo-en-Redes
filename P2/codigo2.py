import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


#1 grafo simple no dirigido aciclico
G=nx.Graph()
G.add_nodes_from(["Bacteria","a", "b", "c","d","e", "f", "g", "h"])
G.add_edges_from([("Bacteria","a"), ("Bacteria","b"),("Bacteria","c"),("Bacteria","d"),
                  ("Bacteria","e"),("Bacteria","f"),("Bacteria","g"),("Bacteria","h")])
node1 = {"Bacteria"}
node2 = {"a", "b", "c","d","e", "f", "g", "h"}


nx.draw(G, with_labels=True, node_size=2000, node_color="aquamarine", pos=nx.circular_layout(G),
        width=3, edge_color='silver', font_size=14)
plt.savefig('grafo11.eps', format='eps', dpi=5000)

nx.draw(G, with_labels=True, node_size=2000, node_color="violet", pos=nx.random_layout(G),
        width=3, edge_color='silver', font_size=14)
plt.savefig('grafo12.eps', format='eps', dpi=5000)

nx.draw(G, with_labels=True, node_size=2000, node_color="lightcoral", pos=nx.spectral_layout(G),
        width=3, edge_color='silver', font_size=14)
plt.savefig('grafo13.eps', format='eps', dpi=5000)

nx.draw(G, with_labels=True, node_size=2000, node_color="yellow", pos=nx.spring_layout(G),
        width=3, edge_color='silver', font_size=14)
plt.savefig('grafo14.eps', format='eps', dpi=5000)

nx.draw(G, with_labels=True, node_size=2000, node_color="orange", pos=nx.shell_layout(G),
        width=3, edge_color='silver', font_size=14)
plt.savefig('grafo15.eps', format='eps', dpi=5000)


nx.draw(G, with_labels=True, node_size=2000, node_color="skyblue", pos=nx.fruchterman_reingold_layout(G),
        width=3, edge_color='silver', font_size=14)
plt.savefig('grafo16.eps', format='eps', dpi=5000)





#2 grafo simple no dirigido ciclico
G=nx.Graph()
G.add_edges_from([(1,2), (2,3), (3,4), (4,5), (5,6), (6,7), (7,1)])
node1 = {1,2,3,4,5,6,7}

fig = plt.figure()
nx.draw(G, with_labels=True, node_size=2000, node_color="aquamarine", pos=nx.circular_layout(G),
        width=3, edge_color='silver', font_size=16)
fig.set_facecolor("blueviolet")
plt.savefig('grafo21.eps', format='eps', dpi=5000)

fig = plt.figure()
nx.draw(G, with_labels=True, node_size=2000, node_color="violet", pos=nx.random_layout(G),
        width=3, edge_color='silver', font_size=16)
fig.set_facecolor("rosybrown")
plt.savefig('grafo22.eps', format='eps', dpi=5000)

fig = plt.figure()
nx.draw(G, with_labels=True, node_size=2000, node_color="lightcoral", pos=nx.spectral_layout(G),
        width=3, edge_color='silver', font_size=16)
fig.set_facecolor("mistyrose")
plt.savefig('grafo23.eps', format='eps', dpi=5000)

fig = plt.figure()
nx.draw(G, with_labels=True, node_size=2000, node_color="yellow", pos=nx.spring_layout(G),
        width=3, edge_color='silver', font_size=16)
fig.set_facecolor("palegreen")
plt.savefig('grafo24.eps', format='eps', dpi=5000)

fig = plt.figure()
nx.draw(G, with_labels=True, node_size=2000, node_color="orange", pos=nx.shell_layout(G),
        width=3, edge_color='silver', font_size=16)
fig.set_facecolor("yellow")
plt.savefig('grafo25.eps', format='eps', dpi=5000)

fig = plt.figure()
nx.draw(G, with_labels=True, node_size=2000, node_color="skyblue", pos=nx.fruchterman_reingold_layout(G),
        width=3, edge_color='silver', font_size=16)
fig.set_facecolor("black")
plt.savefig('grafo26.eps', format='eps', dpi=5000)




#3 grafo simple no dirigido reflexivo
G=nx.Graph()
G.add_nodes_from([1,2,3,4,5,6])
G.add_edges_from([   (1,1) , (2,1) , (3,1) , (4,1) , (5,1) , (6,1),
   (1,2) ,(2,2) ,(3,2) ,(4,2) ,(5,2) ,(6,2) ,
   (1,3) ,(2,3) ,(3,3) ,(4,3) ,(5,3) ,(6,3) ,
   (1,4) ,(2,4) ,(3,4) ,(4,4) ,(5,4) ,(6,4) ,  
   (1,5) ,(2,5) ,(3,5) ,(4,5) ,(5,5) ,(6,5)])

#d h o p s v
color_map = []
for node in G:
    if node >5 :
        color_map.append('magenta')
    else: color_map.append('aquamarine')  
nx.draw_networkx (G, with_labels=True,node_size=700, node_color = color_map,
                  pos=nx.circular_layout(G), node_shape='d', width=3, edge_color='silver', font_size=16)
plt.axis('off')
plt.savefig('grafo31.eps', format='eps', dpi=5000)

color_map = []
for node in G:
    if node >5 :
        color_map.append('yellow')
    else: color_map.append('violet')  
nx.draw_networkx (G, with_labels=True,node_size=700, node_color = color_map,
                  pos=nx.random_layout(G), node_shape='h', width=3, edge_color='silver', font_size=16)
plt.axis('off')
plt.savefig('grafo32.eps', format='eps', dpi=5000)

color_map = []
for node in G:
    if node >5 :
        color_map.append('yellow')
    else: color_map.append('lightcoral')  
nx.draw_networkx (G, with_labels=True,node_size=700, node_color = color_map,
                  pos=nx.spectral_layout(G), node_shape='p', width=3, edge_color='silver', font_size=16)
plt.axis('off')
plt.savefig('grafo33.eps', format='eps', dpi=5000)


color_map = []
for node in G:
    if node >5 :
        color_map.append('chartreuse')
    else: color_map.append('yellow')  
nx.draw_networkx (G, with_labels=True,node_size=700, node_color = color_map,
                  pos=nx.spring_layout(G), node_shape='s', width=3, edge_color='silver', font_size=16)
plt.axis('off')
plt.savefig('grafo34.eps', format='eps', dpi=5000)

color_map = []
for node in G:
    if node >5 :
        color_map.append('yellow')
    else: color_map.append('orange')  
nx.draw_networkx (G, with_labels=True,node_size=700, node_color = color_map,
                  pos=nx.shell_layout(G), node_shape='v', width=3, edge_color='silver', font_size=16)
plt.axis('off')
plt.savefig('grafo35.eps', format='eps', dpi=5000)

color_map = []
for node in G:
    if node >5 :
        color_map.append('yellow')
    else: color_map.append('skyblue')  
nx.draw_networkx (G, with_labels=True,node_size=700, node_color = color_map,
                  pos=nx.fruchterman_reingold_layout(G), node_shape='o', width=3, edge_color='silver', font_size=16)
plt.axis('off')
plt.savefig('grafo36.eps', format='eps', dpi=5000)





#4 grafo simple dirigido aciclico
G=nx.DiGraph()
G.add_edges_from([("Abuelo","Hijo"), ("Abuelo","Hija"), ("Abuela","Hijo"), ("Abuela","Hija"),
                  ("Hija","Nieto")])
    
nx.draw(G, node_size=2000, 
       node_color='white', node_shape='o', width=3, alpha=1, with_labels=None,
       pos=nx.circular_layout(G), font_size=18, edge_color='grey')
plt.axis('off')
plt.savefig('grafo4.jpg', format='jpg', dpi=1000)


G=nx.DiGraph()
G.add_edges_from([("Abuelo","Hijo"), ("Abuelo","Hija"), ("Abuela","Hijo"), ("Abuela","Hija"),
                  ("Hija","Nieto")])
nx.draw(G, node_size=2000, 
       node_color='white', node_shape='o', width=3, alpha=1, with_labels=None,
       pos=nx.spring_layout(G), font_size=18, edge_color='grey')
plt.axis('off')
plt.savefig('grafo4.jpg', format='jpg', dpi=1000)




# 5 grafo simple dirigido ciclico
import networkx as nx
import matplotlib.pyplot as plt


G=nx.DiGraph()
G.add_edges_from([("Evaporacion","Condensacion"), ("Condensacion","Precipitacion"),
                  ("Precipitacion","Infiltracion"), ("Infiltracion","Evaporacion"),
                  ("Infiltracion","Transpiracion"), ("Transpiracion","Condensacion")])

nx.draw(G, node_size=1000, 
       node_color='aquamarine', node_shape='o', width=3, alpha=1, with_labels=True,
       pos=nx.circular_layout(G), font_size=12, edge_color='skyblue', arrows=True)
plt.axis('off')
plt.savefig('grafo51.eps', format='eps', dpi=1000)

nx.draw(G, node_size=1000, 
       node_color='violet', node_shape='o', width=3, alpha=1, with_labels=True,
       pos=nx.random_layout(G), font_size=12, edge_color='skyblue', arrows=True)
plt.axis('off')
plt.savefig('grafo52.eps', format='eps', dpi=1000)

nx.draw(G, node_size=1000, 
       node_color='lightcoral', node_shape='o', width=3, alpha=1, with_labels=True,
       pos=nx.spectral_layout(G), font_size=12, edge_color='skyblue', arrows=True)
plt.axis('off')
plt.savefig('grafo53.eps', format='eps', dpi=1000)

nx.draw(G, node_size=1000, 
       node_color='yellow', node_shape='o', width=3, alpha=1, with_labels=True,
       pos=nx.spring_layout(G), font_size=12, edge_color='skyblue', arrows=True)
plt.axis('off')
plt.savefig('grafo54.eps', format='eps', dpi=1000)

nx.draw(G, node_size=1000, 
       node_color='orange', node_shape='o', width=3, alpha=1, with_labels=True,
       pos=nx.shell_layout(G), font_size=12, edge_color='skyblue', arrows=True)
plt.axis('off')
plt.savefig('grafo55.eps', format='eps', dpi=1000)

nx.draw(G, node_size=1000, 
       node_color='skyblue', node_shape='o', width=3, alpha=1, with_labels=True,
       pos=nx.fruchterman_reingold_layout(G), font_size=12, edge_color='skyblue', arrows=True)
plt.axis('off')
plt.savefig('grafo56.eps', format='eps', dpi=1000)


nx.draw(G, node_size=1000, 
       node_color='skyblue', node_shape='o', width=3, alpha=1, with_labels=True,
       pos=nx.kamada_kawai_layout(G), font_size=12, edge_color='grey', arrows=True)
plt.axis('off')
plt.savefig('grafo57.eps', format='eps', dpi=1000)


#6 grafo simple dirigido reflexivo
G=nx.DiGraph()
G.add_edges_from([   (1,1) , (2,1) , (3,1) , (4,1) , (5,1) , (6,1),
   (1,2) ,(2,2) ,(3,2) ,(4,2) ,(5,2) ,(6,2) ,
   (1,3) ,(2,3) ,(3,3) ,(4,3) ,(5,3) ,(6,3) ,
   (1,4) ,(2,4) ,(3,4) ,(4,4) ,(5,4) ,(6,4) ,  
   (1,5) ,(2,5) ,(3,5) ,(4,5) ,(5,5) ,(6,5) ,
   (1,6) ,(2,6) ,(3,6) ,(4,6) ,(5,6) ,(6,6)])

nx.draw(G, node_size=1000, 
       node_color='violet', node_shape='o', width=3, alpha=1, with_labels=True,
       pos=nx.circular_layout(G), font_size=16, edge_color='silver', arrows=True)
plt.axis('off')
plt.savefig('grafo61.eps', format='eps', dpi=1000)


nx.draw(G, node_size=1000, 
       node_color='aquamarine', node_shape='o', width=3, alpha=1, with_labels=True,
       pos=nx.random_layout(G), font_size=16, edge_color='silver', arrows=True)
plt.axis('off')
plt.savefig('grafo62.eps', format='eps', dpi=1000)

nx.draw(G, node_size=1000, 
       node_color='yellow', node_shape='o', width=3, alpha=1, with_labels=True,
       pos=nx.spectral_layout(G), font_size=16, edge_color='silver', arrows=True)
plt.axis('off')
plt.savefig('grafo63.eps', format='eps', dpi=1000)

nx.draw(G, node_size=1000, 
       node_color='orange', node_shape='o', width=3, alpha=1, with_labels=True,
       pos=nx.spring_layout(G), font_size=16, edge_color='silver', arrows=True)
plt.axis('off')
plt.savefig('grafo64.eps', format='eps', dpi=1000)

nx.draw(G, node_size=1000, 
       node_color='lime', node_shape='o', width=3, alpha=1, with_labels=True,
       pos=nx.shell_layout(G), font_size=16, edge_color='silver', arrows=True)
plt.axis('off')
plt.savefig('grafo65.eps', format='eps', dpi=1000)


nx.draw(G, node_size=1000, 
       node_color='r', node_shape='o', width=3, alpha=1, with_labels=True,
       pos=nx.fruchterman_reingold_layout(G), font_size=16, edge_color='silver', arrows=True)
plt.axis('off')
plt.savefig('grafo66.eps', format='eps', dpi=1000)

nx.draw(G, node_size=1000, 
       node_color='chocolate', node_shape='o', width=3, alpha=1, with_labels=True,
       pos=nx.kamada_kawai_layout(G), font_size=16, edge_color='silver', arrows=True)
plt.axis('off')
plt.savefig('grafo67.eps', format='eps', dpi=1000)




#7 multigrafo no dirigido aciclico
G = nx.MultiGraph()
G.add_edges_from([(1,2) , (2,3) , (3,4)])


pos=nx.circular_layout(G)
nx.draw_networkx_nodes(G, pos, node_color='grey', node_size=600)
nx.draw_networkx_edges(G, pos, edgelist=[(1,2), (2,3)], width=2, edge_color='deepskyblue')
nx.draw_networkx_edges(G, pos, edgelist=[(3, 4)], width=8, edge_color='lime')
nx.draw_networkx_labels(G, pos, font_size=16, font_color='white')
plt.axis('off')
plt.savefig('grafo71.eps', format='eps', dpi=1000)


pos=nx.random_layout(G)
nx.draw_networkx_nodes(G, pos, node_color='grey', node_size=600)
nx.draw_networkx_edges(G, pos, edgelist=[(1,2), (2,3)], width=2, edge_color='deepskyblue')
nx.draw_networkx_edges(G, pos, edgelist=[(3, 4)], width=8, edge_color='lime')
nx.draw_networkx_labels(G, pos, font_size=16, font_color='white')
plt.axis('off')
plt.savefig('grafo72.eps', format='eps', dpi=1000)


pos=nx.spectral_layout(G)
nx.draw_networkx_nodes(G, pos, node_color='grey', node_size=600)
nx.draw_networkx_edges(G, pos, edgelist=[(1,2), (2,3)], width=2, edge_color='deepskyblue')
nx.draw_networkx_edges(G, pos, edgelist=[(3, 4)], width=8, edge_color='lime')
nx.draw_networkx_labels(G, pos, font_size=16, font_color='white')
plt.axis('off')
plt.savefig('grafo73.eps', format='eps', dpi=1000)


pos=nx.spring_layout(G)
nx.draw_networkx_nodes(G, pos, node_color='grey', node_size=600)
nx.draw_networkx_edges(G, pos, edgelist=[(1,2), (2,3)], width=2, edge_color='deepskyblue')
nx.draw_networkx_edges(G, pos, edgelist=[(3, 4)], width=8, edge_color='lime')
nx.draw_networkx_labels(G, pos, font_size=16, font_color='white')
plt.axis('off')
plt.savefig('grafo74.eps', format='eps', dpi=1000)

pos=nx.shell_layout(G)
nx.draw_networkx_nodes(G, pos, node_color='grey', node_size=600)
nx.draw_networkx_edges(G, pos, edgelist=[(1,2), (2,3)], width=2, edge_color='deepskyblue')
nx.draw_networkx_edges(G, pos, edgelist=[(3, 4)], width=8, edge_color='lime')
nx.draw_networkx_labels(G, pos, font_size=16, font_color='white')
plt.axis('off')
plt.savefig('grafo75.eps', format='eps', dpi=1000)


pos=nx.fruchterman_reingold_layout(G)
nx.draw_networkx_nodes(G, pos, node_color='grey', node_size=600)
nx.draw_networkx_edges(G, pos, edgelist=[(1,2), (2,3)], width=2, edge_color='deepskyblue')
nx.draw_networkx_edges(G, pos, edgelist=[(3, 4)], width=8, edge_color='lime')
nx.draw_networkx_labels(G, pos, font_size=16, font_color='white')
plt.axis('off')
plt.savefig('grafo76.eps', format='eps', dpi=1000)





#8 multigrafo no dirigido aciclico

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
img=mpimg.imread('/Users/Serna/Desktop/puente.png')
img2=mpimg.imread('/Users/Serna/Desktop/dospuente.jpg')
# draw graph without images
G =nx.Graph()
G.add_edge(0,1,image=img2,size=0.3)
G.add_edge(0,2,image=img2,size=0.3)
G.add_edge(0,3,image=img,size=0.2)
G.add_edge(1,3,image=img,size=0.2)
G.add_edge(2,3,image=img,size=0.2)

pos=nx.spring_layout(G)

fig = plt.figure()
nx.draw(G, node_size=1000, 
       node_color='peru', node_shape='o', width=3, alpha=1, with_labels=None,edge_style='dashed',
       pos=nx.fruchterman_reingold_layout(G), font_size=16, edge_color='peru', arrows=True)
plt.axis('off')



# add images on edges
ax=plt.gca()
fig=plt.gcf()
label_pos = 0.5 # middle of edge, halfway between nodes
trans = ax.transData.transform
trans2 = fig.transFigure.inverted().transform
imsize = 0.1 # this is the image size
for (n1,n2) in G.edges():
    (x1,y1) = pos[n1]
    (x2,y2) = pos[n2]
    (x,y) = (x1 * label_pos + x2 * (1.0 - label_pos),
             y1 * label_pos + y2 * (1.0 - label_pos))
    xx,yy = trans((x,y)) # figure coordinates
    xa,ya = trans2((xx,yy)) # axes coordinates
    imsize = G[n1][n2]['size']
    img =  G[n1][n2]['image']
    a = plt.axes([xa-imsize/2.0,ya-imsize/2.0, imsize, imsize ])
    a.imshow(img)
    a.set_aspect('equal')
    a.axis('off')
fig.set_facecolor("deepskyblue")







#9 multigrafo no dirigido reflexivo
G=nx.Graph()
G.add_edges_from([   (1,2) ,
   (2,2) ,(3,2) ,(4,2) ,(5,2)  ,
   (2,3) ,(3,3) ,(4,3) ,(5,3)  ,
    (2,4) ,(3,4) ,(4,4) ,(5,4)  ,  
    (2,5) ,(3,5) ,(4,5) ,(5,5) ,(6,5)])
node1 = {2,3,4,5}
node2 = {1}
node3 = {6}

pos=nx.circular_layout(G)
nx.draw_networkx (G, pos, nodelist=node1,node_size=400, node_color='yellow', node_shape='d',
                  font_size=18, edge_color='grey')
nx.draw_networkx (G, pos, nodelist=node2,node_size=400, node_color='lime', node_shape='o',
                  font_size=None, with_labels=False, edge_color='grey')
nx.draw_networkx (G, pos, nodelist=node3,node_size=400, node_color='r', node_shape='o',
                  font_size=None, with_labels=False, edge_color='grey')
plt.axis('off')
plt.savefig('grafo91.eps', format='eps', dpi=1000)



pos=nx.random_layout(G)
nx.draw_networkx (G, pos, nodelist=node1,node_size=400, node_color='yellow', node_shape='d',
                  font_size=18, edge_color='grey')
nx.draw_networkx (G, pos, nodelist=node2,node_size=400, node_color='lime', node_shape='o',
                  font_size=None, with_labels=False, edge_color='grey')
nx.draw_networkx (G, pos, nodelist=node3,node_size=400, node_color='r', node_shape='o',
                  font_size=None, with_labels=False, edge_color='grey')
plt.axis('off')
plt.savefig('grafo92.eps', format='eps', dpi=1000)



pos=nx.spectral_layout(G)
nx.draw_networkx (G, pos, nodelist=node1,node_size=400, node_color='yellow', node_shape='d',
                  font_size=18, edge_color='grey')
nx.draw_networkx (G, pos, nodelist=node2,node_size=400, node_color='lime', node_shape='o',
                  font_size=None, with_labels=False, edge_color='grey')
nx.draw_networkx (G, pos, nodelist=node3,node_size=400, node_color='r', node_shape='o',
                  font_size=None, with_labels=False, edge_color='grey')
plt.axis('off')
plt.savefig('grafo93.eps', format='eps', dpi=1000)



pos=nx.spring_layout(G)
nx.draw_networkx (G, pos, nodelist=node1,node_size=400, node_color='yellow', node_shape='d',
                  font_size=18, edge_color='grey')
nx.draw_networkx (G, pos, nodelist=node2,node_size=400, node_color='lime', node_shape='o',
                  font_size=None, with_labels=False, edge_color='grey')
nx.draw_networkx (G, pos, nodelist=node3,node_size=400, node_color='r', node_shape='o',
                  font_size=None, with_labels=False, edge_color='grey')
plt.axis('off')
plt.savefig('grafo94.eps', format='eps', dpi=1000)



pos=nx.shell_layout(G)
nx.draw_networkx (G, pos, nodelist=node1,node_size=400, node_color='yellow', node_shape='d',
                  font_size=18, edge_color='grey')
nx.draw_networkx (G, pos, nodelist=node2,node_size=400, node_color='lime', node_shape='o',
                  font_size=None, with_labels=False, edge_color='grey')
nx.draw_networkx (G, pos, nodelist=node3,node_size=400, node_color='r', node_shape='o',
                  font_size=None, with_labels=False, edge_color='grey')
plt.axis('off')
plt.savefig('grafo95.eps', format='eps', dpi=1000)


pos=nx.kamada_kawai_layout(G)
nx.draw_networkx (G, pos, nodelist=node1,node_size=400, node_color='yellow', node_shape='d',
                  font_size=18, edge_color='grey')
nx.draw_networkx (G, pos, nodelist=node2,node_size=400, node_color='lime', node_shape='o',
                  font_size=None, with_labels=False, edge_color='grey')
nx.draw_networkx (G, pos, nodelist=node3,node_size=400, node_color='r', node_shape='o',
                  font_size=None, with_labels=False, edge_color='grey')
plt.axis('off')
plt.savefig('grafo96.eps', format='eps', dpi=1000)





#10 Multigrafo dirigido aciclico
G=nx.MultiDiGraph()
G.add_edges_from([   (1,1) , (2,1) , (3,1) , (4,1) , (5,1) ,
   (1,2) ,(2,2) ,(3,2) ,(4,2) ,(5,2)  ,
   (1,3) ,(2,3) ,(3,3) ])

pos=nx.circular_layout(G)
nx.draw_networkx (G, pos, node_size=400, node_color='aquamarine', node_shape='o',
                  font_size=18, edge_color='silver')
plt.axis('off')
plt.savefig('grafo101.eps', format='eps', dpi=1000)


pos=nx.random_layout(G)
nx.draw_networkx (G, pos, node_size=400, node_color='aquamarine', node_shape='o',
                  font_size=18, edge_color='silver')
plt.axis('off')
plt.savefig('grafo102.eps', format='eps', dpi=1000)

pos=nx.spectral_layout(G)
nx.draw_networkx (G, pos, node_size=400, node_color='aquamarine', node_shape='o',
                  font_size=18, edge_color='silver')
plt.axis('off')
plt.savefig('grafo103.eps', format='eps', dpi=1000)

pos=nx.spring_layout(G)
nx.draw_networkx (G, pos, node_size=400, node_color='aquamarine', node_shape='o',
                  font_size=18, edge_color='silver')
plt.axis('off')
plt.savefig('grafo104.eps', format='eps', dpi=1000)

pos=nx.shell_layout(G)
nx.draw_networkx (G, pos, node_size=400, node_color='aquamarine', node_shape='o',
                  font_size=18, edge_color='silver')
plt.axis('off')
plt.savefig('grafo105.eps', format='eps', dpi=1000)

pos=nx.fruchterman_reingold_layout(G)
nx.draw_networkx (G, pos, node_size=400, node_color='aquamarine', node_shape='o',
                  font_size=18, edge_color='silver')
plt.axis('off')
plt.savefig('grafo106.eps', format='eps', dpi=1000)

pos=nx.kamada_kawai_layout(G)
nx.draw_networkx (G, pos, node_size=400, node_color='aquamarine', node_shape='o',
                  font_size=18, edge_color='silver')
plt.axis('off')
plt.savefig('grafo107.eps', format='eps', dpi=1000)





#11 Multigrafo dirigido ciclico
G=nx.MultiDiGraph()
G.add_edges_from([   (1,1) , (2,1) , (3,1) ,
   (1,2)  ,(4,2) ,(5,2) ,(6,2) ,
   (1,3) ,(2,3) ,(3,3) ,(4,3) ,(5,3) ,(6,3) ,
   (1,4) ,(2,4)  ,(6,4) ,  
   (1,5) ,(2,5) ,(3,5) ,(4,5) ,(5,5) ,(6,5) ,
   (1,6)  ,(5,6) ,(6,6)])

pos=nx.circular_layout(G)
nx.draw_networkx (G,node_size=400, node_color='orange', node_shape='o',
                  font_size=18, edge_color='grey')
plt.axis('off')
plt.savefig('grafo111.eps', format='eps', dpi=1000)

pos=nx.random_layout(G)
nx.draw_networkx (G,node_size=400, node_color='violet', node_shape='o',
                  font_size=18, edge_color='grey')
plt.axis('off')
plt.savefig('grafo112.eps', format='eps', dpi=1000)

pos=nx.spectral_layout(G)
nx.draw_networkx (G,node_size=400, node_color='lime', node_shape='o',
                  font_size=18, edge_color='grey')
plt.axis('off')
plt.savefig('grafo113.eps', format='eps', dpi=1000)

pos=nx.kamada_kawai_layout(G)
nx.draw_networkx (G,node_size=400, node_color='aquamarine', node_shape='o',
                  font_size=18, edge_color='grey')
plt.axis('off')
plt.savefig('grafo114.eps', format='eps', dpi=1000)




#12 Multigrafo dirigido reflexivo
G=nx.MultiDiGraph()
G.add_edges_from([   (1,1) , (2,1) , (3,1) ,
   (1,2)  ,(4,2) ,(5,2) ,(6,2) ,
   (1,3) ,(2,3) ,(3,3) ,(4,3) ,(5,3) ,(6,3) ,
   (1,4) ,(2,4)  ,(6,4) ,  
   (1,5) ,(2,5) ,(3,5) ,(4,5) ,(5,5) ,(6,5) ,
   (1,6)  ,(5,6) ,(6,6)])

nx.draw(G, node_size=1000, 
       node_color='white', node_shape='o', width=3, alpha=1, with_labels=None,
       pos=nx.kamada_kawai_layout(G), font_size=16, edge_color='pink', arrows=True)
plt.axis('off')
plt.savefig('grafo121.jpg', format='jpg', dpi=1000)

