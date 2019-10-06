from skimage.morphology import skeletonize
from skimage import data
import numpy as np
import sknw
import matplotlib.pyplot as plt
import show
import spacenetflow as flow
import random

# get satellite image
tb = flow.TargetBundle()
img = show.get_image()

# get target image built from ground truth road coords
tlist = list(tb.targets.values())
t = tlist[random.randint(0, len(tlist))]
timg = t.image()[:,:,0]

# skeletonize target image
simg = skeletonize(timg)

# build graph from skeleton
graph = sknw.build_sknw(simg)

# draw them all
fig = plt.figure()
fig.add_subplot(1,4,1)
fig = plt.figure()
plt.imshow(img)
plt.title("Satellite image")

fig.add_subplot(1,4,2)
plt.imshow(timg)
plt.title("Target image")

fig.add_subplot(1,4,3)
plt.imshow(simg)
plt.title("Skeleton image")

fig.add_subplot(1,4,4)
plt.imshow(simg)
plt.title("Skeleton image w/ points")

# draw edges by pts
for (s,e) in graph.edges():
    ps = graph[s][e]['pts']
    plt.plot(ps[:,1], ps[:,0], 'green')
    
# draw node by o
node, nodes = graph.node, graph.nodes()
ps = np.array([node[i]['o'] for i in nodes])
plt.plot(ps[:,1], ps[:,0], 'r.')

# title and show
plt.show()
