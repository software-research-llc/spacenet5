import os
import sys
import logging
import tqdm
import numpy as np
import snflow as flow
import networkx as nx
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


def rotation_matrix(theta):
    return np.array([[ np.cos(theta), 0 - np.sin(theta), 0],
                     [ np.sin(theta), np.cos(theta), 0],
                     [ 0, 0, 1]])

def rotate(x, y, angle):
    theta = angle / 2 * np.pi
    mat = np.array([x,y,0])
    return np.matmul(rotation_matrix(theta), mat)

def tr_x(x):
    x = x * flow.CHIP_CANVAS_SIZE[0] / flow.TARGET_IMSHAPE[0]
    return x

def tr_y(y):
    y = y * flow.CHIP_CANVAS_SIZE[1] / flow.TARGET_IMSHAPE[1]
    return y

def get_coords(graph, cn=255, ce=128):
    acc = np.cumprod((1,)+tuple([256,256])[::-1][:-1])[::-1]
    img = np.zeros([flow.TARGET_IMSHAPE[0], flow.TARGET_IMSHAPE[1]])
    img = img.ravel()
    weights = []

    for idx in graph.nodes():
        pts = graph.node[idx]['pts']
        img[np.dot(pts,acc)] = cn
    for (s,e) in graph.edges():
        eds = graph[s][e]
        pts = eds['pts']
        weights.append(eds['weight'])
        img[np.dot(pts,acc)] = ce

    img = img.reshape((256,256))
    points = []
    for x in range(256):
        for y in range(256):
            if img[x][y] == ce:
                points.append((x,y))
    return points, weights

def graphs_to_wkt(graphs, output_csv_path):
    output_csv = open(output_csv_path, "w")
    output_csv.write("ImageId,WKT_Pix,length_m,time_s\n")

    for graph in tqdm.tqdm(graphs):
        log.debug("Converting {} to WKT...".format(graph.name))
        chipname = os.path.basename(graph.name).replace("PS-RGB_", "").replace(".tif", "")#os.path.basename(filename).replace("PS-RGB_", "").replace(".tif", "")
        linestrings = []
        weights = []
        seen = set()

        for (s,e) in graph.edges():
            if (s,e) in seen:
                continue
            elif (e,s) in seen:
                continue
            else:
                seen.add((s,e))
                seen.add((e,s))

            weight = graph[s][e]['weight']
            weights.append(weight)

            pts = graph[s][e]['pts']
            xs = pts[:,1]
            ys = pts[:,0]
            if len(xs) > 1 and len(ys) > 1:
                linestring = "LINESTRING ({} {}".format(tr_x(xs[0]), tr_y(ys[0]))
                for i in range(1, len(pts)):
                    linestring += ", {} {}".format(tr_x(xs[i]), tr_y(ys[i]))
                linestring += ")"
                log.debug(linestring)
                linestrings.append(linestring)
            else:
                log.info("{}: unconnected point".format(chipname))
#            import pdb; pdb.set_trace()

        for idx,linestring in enumerate(linestrings):
            error_shown = False
            output_csv.write("{},".format(chipname))
            output_csv.write('"{}",'.format(linestring))
            output_csv.write("0.0,")
            try:
                output_csv.write("{}\n".format(weights[i]))
            except Exception as exc:
                if not error_shown:
                    log.error("{} at idx {} / {} in {}".format(str(exc), idx, len(weights), chipname))
                output_csv.write("0.0\n")
        """
            print("weight:")
            print(weight)
            print("xs:")
            print(xs)
            print("ys:")
            print(ys)
            print("------")
            print("graph[s][e]:")
            print(graph[s][e])
            ps = graph[s][e]['pts']
            points = (ps[:,1], ps[:,0])
            print("ps:")
            print(ps)
            print("points:")
            print(points)
            node, nodes = graph.node, graph.nodes()
            ps = np.array([node[i]['o'] for i in nodes])
            print(ps[:,1], ps[:,0])
        """
    output_csv.close()
    sys.exit()
