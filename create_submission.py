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
    """Translate resized target image coordinates to input image coordinates"""
    x = x * flow.ORIG_IMSHAPE[0] / flow.IMSHAPE[0]
    return x

def tr_y(y):
    """Translate resized target image coordinates to input image coordinates"""
    y = y * flow.ORIG_IMSHAPE[1] / flow.IMSHAPE[1]
    return y

def get_coords(graph, cn=255, ce=128):
    acc = np.cumprod((1,)+tuple([256,256])[::-1][:-1])[::-1]
    img = np.zeros([flow.IMSHAPE[0], flow.IMSHAPE[1]])
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

def get_weight_for(mask, graph, cp):
    """Return the average speed, judged by pixel intensity and color,
       for the given points.  Ret val should be multiplied by path length."""
    max_speed = 29
    xs = cp[:,1]
    ys = cp[:,0]

    weight = 0
    for x,y in zip(xs,ys):
        channels = mask[int(round(x))][int(round(y))]
        for i in range(1, mask.shape[-1]):
            # channel[0] * 3, channel[1] * 2, channel[2] * 1, etc.
            if channels[i] > 1.0:
                log.critical("expected range [0.0,1.0], got {}".format(channels[i]))
            weight += (max_speed - flow.get_channel_speed(i)) * channels[i]
        weight = weight / 3
    if weight <= 0:
        weight = 1
    assert weight > 0, "did not detect the color of a path for {}".format(graph.name)

    return weight

def get_weight_for2(mask, graph, xy1, xy2):
    weight = 0
    max_speed = 29
    x1,y1 = xy1
    x2,y2 = xy2
    x1,y1 = int(round(x1)), int(round(y1))
    x2,y2 = int(round(x2)), int(round(y2))
    for i in range(1, mask.shape[-1]):
        channel_speed = flow.get_channel_speed(i)
        weight += mask[x1,y1,i] * channel_speed
        mean = np.mean(mask[x1:x2,y1:y2,i])
        if np.isfinite(mean):
            weight += mean * channel_speed
    return weight

def get_weight_for3(mask, x, y):
    weight = 0
    x,y = int(x), int(y)
    for i in range(1,mask.shape[-1]):
        try:
            speed = mask[x,y,i] * flow.get_channel_speed(i)
        except Exception as exc:
            print("\nx,y,i: {},{},{}".format(x,y,i))
            raise exc
        if speed <= 0:
            speed = flow.get_channel_speed(i)
        weight += speed
    return weight

def slope(x1,y1,x2,y2):
    return (y2 - y1) / (x2 - x1)

def graphs_to_wkt(masks, graphs, output_csv_path):
    output_csv = open(output_csv_path, "w")
    output_csv.write("ImageId,WKT_Pix,length_m,time_s\n")

    for mask, graph in tqdm.tqdm(zip(masks, graphs)):
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

#            weight = graph[s][e]['weight']
#            weights.append(weight)

            pts = [graph.node[s]['o'].tolist()][::-1]
            pts += graph[s][e]['pts'].tolist()
            pts += [graph.node[e]['o'].tolist()][::-1]
            pts = np.array(pts)
            xs = pts[:,1]
            ys = pts[:,0]
            
#            node, nodes = graph.node, graph.nodes()
#            ps = np.array([node[i]['o'] for i in nodes])
#            for (x, y) in (ps[:,1], ps[:,0]):
#                pts.append([x,y])
#            import pdb; pdb.set_trace()
#            node, nodes = graph.node, graph.nodes()
#            cps = [node[i]['o'] for i in nodes]
            # add nodes
            """
            for xy in graph.node[s]['pts']:
                if len(xy) < 4:
                    continue
                linestring = "LINESTRING ({} {}".format(tr_x(xy[0][0]), tr_y(xy[0][1]))
                for x,y in xy[1:]:
                    linestring += ", {} {}".format(tr_x(x), tr_y(y))
                linestring += ")"
                linestrings.append(linestring)
                weights.append(0.01)
            for xy in graph.node[s]['pts']:
                if len(xy) < 4:
                    continue
                linestring = "LINESTRING ({} {}".format(tr_x(xy[0][0]), tr_y(xy[0][1]))
                for x,y in xy[1:]:
                    linestring += ", {} {}".format(tr_x(x), tr_y(y))
                linestring += ")"
                linestrings.append(linestring)
                weights.append(0.01)
            """
            # add edges
            weight = 0
            if len(xs) > 1:
                # calculate time to travel this edge

                for i in range(1, len(xs), 2):
                    p1,q1 = tr_x(xs[i-1]), tr_y(ys[i-1])
                    p2,q2 = tr_x(xs[i]), tr_y(ys[i])
                    # Euclidean distance
#                    euclid_distance = np.sqrt( (p2 - p1) * (p2 - p1) + (q2 - q1) * (q2 - q1) )

                # construct linestring
                linestring = "LINESTRING ({} {}".format(tr_x(xs[0]), tr_y(ys[0]))
                for i in range(1, len(pts)):
                    linestring += ", {} {}".format(tr_x(xs[i]), tr_y(ys[i]))
                    weight += get_weight_for3(mask, xs[i], ys[i])
                linestring += ")"
                log.debug(linestring)
               
                # skip cycles starting and ending at the same node
                if xs[0] == xs[-1] and ys[0] == ys[-1]:
                    continue
                weights.append(weight)
                linestrings.append(linestring)
            else:
                log.warning("{}: unconnected point {}".format(chipname, str(xs)))
#        import pdb; pdb.set_trace()
        if len(linestrings) < 1:
            output_csv.write("{},".format(chipname))
            output_csv.write('"LINESTRING EMPTY",')
            output_csv.write("0.0,0.0\n")

        for idx,linestring in enumerate(linestrings):
            error_shown = False
            output_csv.write("{},".format(chipname))
            output_csv.write('"{}",'.format(linestring))
            output_csv.write("0.0,")
            try:
                if float("{:10.10f}".format(weights[idx])) == 0:
                    output_csv.write("1.0\n")
                else:
                    output_csv.write("{:10.10f}\n".format(weights[idx]))
            except Exception as exc:
                if not error_shown:
                    error_shown = True
                    log.error("{} at idx {} / {} in {}".format(str(exc), idx, len(weights), chipname))
                else:
                    log.debug("{} at idx {} / {} in {}".format(str(exc), idx, len(weights), chipname))
                output_csv.write("0.0\n")

    output_csv.close()
