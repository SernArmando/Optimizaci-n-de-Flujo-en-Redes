import networkx as nx
import matplotlib.pyplot as plt

#1 grafo simple no dirigido aciclico
G=nx.Graph()
G.add_edges_from([("Bacteria","a"), ("Bacteria","b"),("Bacteria","c"),("Bacteria","d"),("Bacteria","e"),("Bacteria","f"),("Bacteria","g"),("Bacteria","h")])
node1 = {"Bacteria"}
node2 = {"a", "b", "c","d","e", "f", "g", "h"}

pos = {"Bacteria":(0, 0),"a":(0,5), "b":(-5,5), "c":(-5,0), "d":(-5,-5), "e":(0,-5), "f":(5,-5), "g":(5,0), "h":(5,5)}

nx.draw_networkx_nodes(G, pos, nodelist=node1,node_size=5000, 
                       node_color='green', node_shape='o', width=5, alpha=1)

nx.draw_networkx_nodes(G, pos, nodelist=node2,node_size=400, 
                       node_color='green', node_shape='o', width=5, alpha=1)

nx.draw_networkx_edges(G, pos, width=3, alpha=0.5, edge_color='grey')

nx.draw_networkx_labels(G, pos, font_size=18)
plt.axis('off')
plt.savefig('grafo1.eps', format='eps', dpi=1000)

#grafo simple no dirigido ciclico
G=nx.Graph()
G.add_edges_from([(1,2), (2,3), (3,4), (4,5), (5,6), (6,7), (7,1)])
node1 = {1,2,3,4,5,6,7}

nx.draw(G, nodelist=node1,node_size=800, 
       node_color='yellow', node_shape='o', width=5, alpha=1,
       with_labels=True, font_size=18, edge_color='grey')

plt.axis('off')
plt.savefig('grafo2.eps', format='eps', dpi=1000)

#grafo simple no dirigido reflexivo
G=nx.Graph()
G.add_edges_from([   (1,1) , (2,1) , (3,1) , (4,1) , (5,1) , (6,1),
   (1,2) ,(2,2) ,(3,2) ,(4,2) ,(5,2) ,(6,2) ,
   (1,3) ,(2,3) ,(3,3) ,(4,3) ,(5,3) ,(6,3) ,
   (1,4) ,(2,4) ,(3,4) ,(4,4) ,(5,4) ,(6,4) ,  
   (1,5) ,(2,5) ,(3,5) ,(4,5) ,(5,5) ,(6,5)])
node1 = {1,2,3,4,5}
node2 = {6}

nx.draw_networkx (G, nodelist=node1,node_size=400, node_color='r', node_shape='o',
                  font_size=18, edge_color='grey')
plt.axis('off')
plt.savefig('grafo3.eps', format='eps', dpi=1000)


#grafo simple dirigido aciclico
G=nx.DiGraph()
G.add_edges_from([("Abuelo","Hijo"), ("Abuelo","Hija"), ("Abuela","Hijo"), ("Abuela","Hija"),
                  ("Hija","Nieto")])

nx.draw(G, node_size=2000, 
       node_color='yellow', node_shape='o', width=3, alpha=1,
       with_labels=True, font_size=18, edge_color='grey')

plt.axis('off')
plt.savefig('grafo4.eps', format='eps', dpi=1000)

#grafo simple dirigido reflexivo
G=nx.DiGraph()
G.add_edges_from([   (1,1) , (2,1) , (3,1) , (4,1) , (5,1) , (6,1),
   (1,2) ,(2,2) ,(3,2) ,(4,2) ,(5,2) ,(6,2) ,
   (1,3) ,(2,3) ,(3,3) ,(4,3) ,(5,3) ,(6,3) ,
   (1,4) ,(2,4) ,(3,4) ,(4,4) ,(5,4) ,(6,4) ,  
   (1,5) ,(2,5) ,(3,5) ,(4,5) ,(5,5) ,(6,5) ,
   (1,6) ,(2,6) ,(3,6) ,(4,6) ,(5,6) ,(6,6)])

nx.draw_networkx (G,node_size=400, node_color='purple', node_shape='o',
                  font_size=18, edge_color='grey')
plt.axis('off')
plt.savefig('grafo6.eps', format='eps', dpi=1000)


#multigrafo no dirigido aciclico
G = nx.MultiGraph()
pos = {4:(-1, 0), 5:(0,1), 6:(0,-1), 7:(1,0)}

nx.draw_networkx_nodes(G, pos,
                       nodelist=[4, 5, 6, 7],
                       node_color='grey',
                       node_size=600,
                       alpha=1)

nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
nx.draw_networkx_edges(G, pos,
                       edgelist=[(7,5), (6,7)],
                       width=2, alpha=0.5, edge_color='blue')
nx.draw_networkx_edges(G, pos,
                       edgelist=[(4, 7)],
                       width=2, alpha=0.5, edge_color='blue')
nx.draw_networkx_edges(G, pos,
                       edgelist=[(4, 6), (4, 5)],
                       width=8, alpha=0.5, edge_color='green')
nx.draw_networkx_labels(G, pos)

plt.axis('off')
plt.savefig('grafo8.eps', format='eps', dpi=1000)

#multigrafo no dirigido reflexivo
G=nx.Graph()
G.add_edges_from([   (1,2) ,
   (2,2) ,(3,2) ,(4,2) ,(5,2)  ,
   (2,3) ,(3,3) ,(4,3) ,(5,3)  ,
    (2,4) ,(3,4) ,(4,4) ,(5,4)  ,  
    (2,5) ,(3,5) ,(4,5) ,(5,5) ,(6,5)])
node1 = {2,3,4,5}
node2 = {1,6}

nx.draw_networkx (G, nodelist=node1,node_size=400, node_color='blue', node_shape='o',
                  font_size=18, edge_color='grey')
nx.draw_networkx (G, nodelist=node2,node_size=400, node_color='white', node_shape='o',
                  font_size=None, with_labels=False, edge_color='white')
plt.axis('off')
plt.savefig('grafo9.eps', format='eps', dpi=1000)

#
G=nx.DiGraph()
G.add_edges_from([("Abuelo","Hijo"), ("Abuelo","Hija"), ("Abuela","Hijo"), ("Abuela","Hija"),
                  ("Hija","Nieto")])

nx.draw(G, node_size=2000, 
       node_color='yellow', node_shape='o', width=3, alpha=1,
       with_labels=True, font_size=18, edge_color='grey')

plt.axis('off')
plt.savefig('grafo4.eps', format='eps', dpi=1000)

#multigrafo dirigido reflexivo
G=nx.DiGraph()
G.add_edges_from([   (1,1) , (2,1) , (3,1) , (4,1) , (5,1) , (6,1),
   (1,2) ,(2,2) ,(3,2) ,(4,2) ,(5,2) ,(6,2) ,
   (1,3) ,(2,3) ,(3,3) ,(4,3) ,(5,3) ,(6,3) ,
   (1,4) ,(2,4) ,(3,4) ,(4,4) ,(5,4) ,(6,4) ,  
   (1,5) ,(2,5) ,(3,5) ,(4,5) ,(5,5) ,(6,5) ,
   (1,6) ,(2,6) ,(3,6) ,(4,6) ,(5,6) ,(6,6)])

nx.draw_networkx (G,node_size=400, node_color='blue', node_shape='o',
                  font_size=18, edge_color='grey')
plt.axis('off')
plt.savefig('grafo10.eps', format='eps', dpi=1000)


#multigrafo dirigido ciclico
G=nx.DiGraph()
G.add_edges_from([   (1,1) , (2,1) , (3,1) , (4,1) , (5,1) ,
   (1,2) ,(2,2) ,(3,2) ,(4,2) ,(5,2)  ,
   (1,3) ,(2,3) ,(3,3) ])

nx.draw_networkx (G,node_size=400, node_color='green', node_shape='o',
                  font_size=18, edge_color='red')
plt.axis('off')
plt.savefig('grafo11.eps', format='eps', dpi=1000)


###########
G=nx.DiGraph()
G.add_edges_from([   (1,1) , (2,1) , (3,1) ,
   (1,2)  ,(4,2) ,(5,2) ,(6,2) ,
   (1,3) ,(2,3) ,(3,3) ,(4,3) ,(5,3) ,(6,3) ,
   (1,4) ,(2,4)  ,(6,4) ,  
   (1,5) ,(2,5) ,(3,5) ,(4,5) ,(5,5) ,(6,5) ,
   (1,6)  ,(5,6) ,(6,6)])

nx.draw_networkx (G,node_size=400, node_color='orange', node_shape='o',
                  font_size=18, edge_color='grey')
plt.axis('off')
plt.savefig('grafo12.eps', format='eps', dpi=1000)