import sys
import snflow as flow
import numpy as np
import snmodel
import cv2
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt
import sknw
import keras
import time
import getch
import os

def infer_mask(model, image):
    if len(image.shape) == 3:
        image = image.reshape([1,] + flow.IMSHAPE)
    output = model.predict(image)
    return np.array(output)

def infer_roads(masks, chipname=''):
    skels = []
    graphs = []
    for mask in masks:
        img = prep_for_skeletonize(mask)
        skel = skeletonize(img)
        graph = sknw.build_sknw(skel)
        graph.name = chipname
        return graph, img, skel

def prep_for_skeletonize(img):
#    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
#    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#    img = dilate_and_erode(img)
    _, img = cv2.threshold(img, 0.15, 1, cv2.THRESH_TRUNC)
    img = np.array(np.ceil(img), dtype=np.float64)
    return img.reshape(flow.TARGET_IMSHAPE[0], flow.TARGET_IMSHAPE[1])

def dilate_and_erode(img):
    kernel = np.ones((1,15))
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)
    return img

def infer(model, image, chipname=''):
    if len(image.shape) == 4:
        mask = infer_mask(model, image)
    else:
        mask = infer_mask(model, image.reshape([1,] + flow.IMSHAPE))
    graph, preproc, skel = infer_roads(mask, chipname)
    return mask, graph, preproc, skel

def infer_and_show(model, image, filename):
    tb = flow.TargetBundle()
    chipid = os.path.basename(filename)#.replace("PS-RGB_", "").replace(".tif", "")
    if isinstance(image, str):
        image = flow.get_image(image)
#    mask, graph, preproc, skel = infer(model, image)
    mask = infer_mask(model, image)
    dilated_mask = dilate_and_erode(mask[0][:,:,0])
#    import pdb;pdb.set_trace()
    for idx in range(1):
        fig = plt.figure()
        fig.add_subplot(2,3,1)
        plt.axis('off')
        plt.imshow(image)
        plt.title("1. Input (satellite image)")

        fig.add_subplot(2,3,2)
        plt.axis('off')
        plt.imshow(mask[0][:,:,0])
        plt.title("2. Mask (network output)")

        fig.add_subplot(2,3,3)
        plt.axis('off')
        plt.imshow(tb[chipid].image()[:,:,0])
        plt.title("Ground truth")
        plt.show()
        return
        """
        fig.add_subplot(2,3,3)
        plt.axis('off')
        plt.imshow(dilated_mask)
        plt.title("3. Dilated + eroded")
        """
        fig.add_subplot(2,3,4)
        plt.axis('off')
        plt.imshow(preproc)
        plt.title("4. Preprocessed mask")

        fig.add_subplot(2,3,5)
        plt.axis('off')
        plt.imshow(skel)
        plt.title("5. Skeletonized mask")

        fig.add_subplot(2,3,6)
        plt.axis('off')
        plt.imshow(skel)
        plt.title("6. Resulting graph")
        for (s,e) in graph.edges():
            ps = graph[s][e]['pts']
            plt.plot(ps[:,1], ps[:,0], 'blue')
    
        node, nodes = graph.node, graph.nodes()
        ps = np.array([node[i]['o'] for i in nodes])
        try:
            plt.plot(ps[:,1], ps[:,0], 'r+')
        except IndexError as exc:
            print("WARNING: IndexError: %s" % exc)

        plt.show()

def do_all(loop=True):
    model = keras.models.load_model("model.tf")
    while True:
        path = flow.get_file()
        image = flow.resize(flow.get_image(path), flow.IMSHAPE).reshape(flow.IMSHAPE)
        if not loop:
            ret = infer(model, image) + (path,)
            return ret
        infer_and_show(model, image, path)
        time.sleep(1)

if __name__ == "__main__":
    do_all()
