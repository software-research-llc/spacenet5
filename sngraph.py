import snflow as flow
import networkx as nx

class SNGraph(nx.Graph):
    def add_channel(self, img):
        img = img.squeeze()
        img = img.clip(0.25, 1)
        assert len(img.shape) == 2, "expected shape (H, W), got {}".format(str(img.shape))
        for x in range(img.shape[0]):
            for y in range(img.shape[1]):
                if img[x][y] > 0:
                    self.insertEdgesFor(img, x, y)

    def insertEdgesFor(self, img, x, y):
        for i in [x, x+1]:
            for j in [y, y+1]:
                if i >= len(img) or j >= len(img[i]):
                    continue
                if img[i][j] > 0:
                    if (i,j) != (x,y):
                        self.add_node((i,j))
                        self.add_edge((x,y), (i,j), weight=img[x][y])

    def add_image(self, img):
        img = img.squeeze()
        assert len(img.shape) == 3, "expected shape (H, W, 4), got {}".format(str(img.shape))
        for i in range(3):
            self.add_channel(img[:,:,i+1])

    def _tr(self, x, y):
        return flow.trup(x, y)

    def toWKT(self):
        seen = set()
        linestrings = []
        for (s,e) in self.edges():
            if (s,e) in seen:
                continue
            elif (e,s) in seen:
                continue
            else:
                seen.add((s,e))
                seen.add((e,s))

            # linestring here refers to the entire corresponding line of the output
            x1, y1 = s
            x2, y2 = e
            linestring = self.name + ',' + '"LINESTRING ({} {}'.format(*self._tr(x1, y1))
            linestring += ', {} {})"'.format(*self._tr(x2, y2))
            linestring += ',0,{}'.format(self[(x1,y1)][(x2,y2)]['weight'])
            linestrings.append(linestring + "\n")
        return linestrings

if __name__ == '__main__':
    import time
    seq = flow.SpacenetSequence.all(batch_size=1)
    x,y = seq[0]
    graph = SNGraph(name=seq.y[0].imageid)
    print("Processing...")
    start = time.time()
    graph.add_image(y)
    wkt = graph.toWKT()
    end = time.time()
    for line in wkt[:5]:
        print(line)
    print("Number of linestrings: {}\nWall time: {}".format(len(wkt), end - start))
