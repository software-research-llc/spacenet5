from skimage.morphology import skeletonize
from skimage import data
import numpy as np
import sknw
import matplotlib.pyplot as plt
import show
import snflow as flow
import random

# get target image built from ground truth road coords
tb = flow.TargetBundle()
tlist = list(tb.targets.values())
while True:
    t = tlist[random.randint(0, len(tlist))]
    timg = t.image()[:,:,0]
    img = flow.get_image(t.chip())
    if t is None or img is None:
        raise Exception("ImageID not found: %s" % path)

    # skeletonize target image
    simg = skeletonize(timg)

    # build graph from skeleton
    graph = sknw.build_sknw(simg, True)

    # draw the sattelite image
    fig = plt.figure()
    fig.add_subplot(1,4,1)
    plt.imshow(img)
    plt.title(t.imageid)

    # draw the target
    fig.add_subplot(1,4,2)
    plt.imshow(timg)
    plt.title("Target image")
   
    # draw the skeleton image
    fig.add_subplot(1,4,3)
    plt.imshow(simg)
    plt.title("Skeleton image")

    # draw the skeleton w/ nodes
    fig.add_subplot(1,4,4)
    plt.imshow(simg)
    plt.title("Nodes/edges")

    # draw edges by pts
    for (s,e) in graph.edges():
        ps = graph[s][e][0]['pts']
        plt.plot(ps[:,1], ps[:,0], 'green')
    
    # draw node by o
    node, nodes = graph.node, graph.nodes()
    ps = np.array([node[i]['o'] for i in nodes])
    plt.plot(ps[:,1], ps[:,0], 'r.')

    # title and show
    print(t.imageid, t.chip())
    plt.show()
