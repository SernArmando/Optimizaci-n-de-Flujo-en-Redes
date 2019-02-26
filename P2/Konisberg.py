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
    