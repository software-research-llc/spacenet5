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
import skimage
import copy

def infer_mask(model, image):
    assert image.ndim == 3, "expected shape {}, got {}".format(flow.IMSHAPE, image.shape)
    output = model.predict(image.reshape([1] + flow.IMSHAPE))
    return np.array(output)[0]

def infer_roads(mask, chipname=''):
    assert mask.ndim == 3, "expected shape {}, got {}".format(flow.TARGET_IMSHAPE, mask.shape)
    img = prep_for_skeletonize(mask)
    skel = skeletonize(img)
    graph = sknw.build_sknw(skel)
    graph.name = chipname
    assert np.max(img) <= 1
    assert np.max(skel) <= 1
    return graph, img, skel

def prep_for_skeletonize(img):
#    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
#    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#    img = dilate_and_erode(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    assert np.max(img) <= 1, "expected np.max(img) == 1, but is {}".format(np.max(img))
    _, img = cv2.threshold(img, 0.25, 1, cv2.THRESH_BINARY)
    return img

def dilate_and_erode(img):
    kernel = np.ones((1,15))
#    kernel = np.array([[1, 0, 1, 0, 1],
#                       [0, 1, 9, 1, 0],
#                       [1, 0, 1, 0, 1]], dtype = np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)
    return img

def infer(model, image, chipname=''):
    assert image.ndim == 3, "expected shape {}, got {}".format(flow.IMSHAPE, image.shape)
    mask = infer_mask(model, image)
    graph, preproc, skel = infer_roads(mask, chipname)
    return mask, graph, preproc, skel

def infer_and_show(model, image, filename):
    tb = flow.TargetBundle()
    chipid = os.path.basename(filename)#.replace("PS-RGB_", "").replace(".tif", "")
    if isinstance(image, str):
        image = flow.get_image(image)
    mask, graph, preproc, skel = infer(model, image)
#    mask = infer_mask(model, image)
#    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    dilated_mask = dilate_and_erode(mask)
#    import pdb;pdb.set_trace()
    for idx in range(1):
        fig = plt.figure()
        fig.add_subplot(2,4,1)
        plt.axis('off')
        plt.imshow(image)
        plt.title("1. Satellite image (neural net input)")

        fig.add_subplot(2,4,2)
        plt.axis('off')
        plt.imshow(tb[chipid].image())
        plt.title("2. Ground truth")

        fig.add_subplot(2,4,3)
        plt.axis('off')
        plt.imshow(mask)
        plt.title("3. Mask (neural net output)")

        fig.add_subplot(2,4,4)
        plt.axis('off')
        plt.imshow(dilated_mask)
        plt.title("4. Dilated + eroded")

        fig.add_subplot(2,4,5)
        plt.axis('off')
        plt.imshow(preproc)
        plt.title("5. Preprocessed mask")

        fig.add_subplot(2,4,6)
        plt.axis('off')
        plt.imshow(skel)
        plt.title("6. Skeletonized mask")

        fig.add_subplot(2,4,7)
        plt.axis('off')
        plt.imshow(skel)
        plt.title("7. Resulting graph")
        for (s,e) in graph.edges():
            ps = graph[s][e]['pts']
            plt.plot(ps[:,1], ps[:,0], 'blue')
    
        node, nodes = graph.node, graph.nodes()
        ps = np.array([node[i]['o'] for i in nodes])
        try:
            plt.plot(ps[:,1], ps[:,0], 'r+')
        except IndexError as exc:
            print("WARNING: IndexError: %s" % exc)

#        import pdb; pdb.set_trace()
        plt.show()

def do_all(loop=True):
    model = snmodel.load_model()
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
