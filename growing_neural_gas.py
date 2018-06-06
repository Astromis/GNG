# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""Initialize module utils."""

from math import sqrt
from mayavi import mlab
import imageio
from collections import OrderedDict
from scipy.spatial.distance import euclidean
import csv
import numpy as np
import networkx as nx
import re
import os
import shutil
import glob
from past.builtins import xrange
from future.utils import iteritems


def shrink_to_3d(data):
    result = []
    
    for i in data:
        depth = len(i)
        if depth <=3:
            result.append(i)
        else:
            sm = np.sum(i)
            sm = depth - sm
            if sm == 0:
                sm = 1

            i0 = i[0]
            i1 = i[1]
            i2 = i[2]
            result.append([i0 * depth / sm, i1 * depth / sm, i2 * depth / sm])
            
    return np.array(result)


def read_test_file():
    """Read the file and return the indices as list of lists."""
    filename = 'test_data.txt'
    with open(filename) as file:
        array2d = [[int(digit) for digit in line.split()] for line in file]

    return array2d


def create_test_data_graph(array2d):
    """Create the graph and returns the networkx version of it 'G'."""

    pos = None

    row, column = len(array2d), len(array2d[0])
    count = 0

    G = nx.Graph()

    for j in xrange(column):
        for i in xrange(row):
            if array2d[row - 1 - i][j] == 0:
                G.add_node(count, pos=(j, i, np.random.randint(-1, 1)))
                count += 1

    pos = nx.get_node_attributes(G, 'pos')

    """
    for index in pos.keys():
        for index2 in pos.keys():
            if pos[index][0] == pos[index2][0] and pos[index][1] == pos[index2][1] - 1:
                G.add_edge(index, index2, weight=1)
            if pos[index][1] == pos[index2][1] and pos[index][0] == pos[index2][0] - 1:
                G.add_edge(index, index2, weight=1)
    """
    return G, pos


def create_data_graph(dots):
    """Create the graph and returns the networkx version of it 'G'."""

    pos = None
    count = 0

    G = nx.Graph()

    for i in dots:
        G.add_node(count, pos=(i))
        count += 1

    pos = nx.get_node_attributes(G, 'pos')

    """
    for index in pos.keys():
        for index2 in pos.keys():
            if pos[index][0] == pos[index2][0] and pos[index][1] == pos[index2][1] - 1:
                G.add_edge(index, index2, weight=1)
            if pos[index][1] == pos[index2][1] and pos[index][0] == pos[index2][0] - 1:
                G.add_edge(index, index2, weight=1)
    """
    return G, pos


def draw_graph3d(graph, fignum, clear=True, size=(1024, 768), graph_colormap='viridis', bgcolor = (1, 1, 1),
                 node_color=(0.3, 0.65, 0.3), node_size=0.3,
                 edge_color=(0.3, 0.3, 0.9), edge_size=0.05,
                 text_size=0.008, text_color=(0, 0, 0)):

    # https://stackoverflow.com/questions/17751552/drawing-multiplex-graphs-with-networkx
    gr = graph #nx.convert_node_labels_to_integers(graph)
    mlab.options.offscreen = True
    graph_pos = nx.get_node_attributes(graph, 'pos')
    
    # numpy array of x, y, z positions in sorted node order
    xyz = shrink_to_3d(np.array([graph_pos[v] for v in sorted(gr)]))
    
    # scalar colors
    scalars = np.array([n for n in gr.nodes()])
    
    if mlab.options.offscreen:
        mlab.figure(fignum, bgcolor=bgcolor, size=size)
    else:
        if fignum == 0:
            mlab.figure(fignum, bgcolor=bgcolor, size=size)

    if clear:
        mlab.clf()

    #----------------------------------------------------------------------------
    # the x,y, and z co-ordinates are here
    # manipulate them to obtain the desired projection perspective 

    pts = mlab.points3d(xyz[:,0], xyz[:,1], xyz[:,2],
                        # scalars,
                        scale_factor=node_size,
                        scale_mode='none',
                        color=node_color,
                        colormap=graph_colormap,
                        resolution=20,
                        transparent=False)
    #----------------------------------------------------------------------------
    """
    for i, (x, y, z) in enumerate(xyz):
        label = mlab.text(x, y, str(i), z=z,
                          width=text_size, name=str(i), color=text_color)
        label.property.shadow = True
    """
    pts.mlab_source.dataset.lines = np.array([e for e in gr.edges()])
    tube = mlab.pipeline.tube(pts, tube_radius=edge_size)
    mlab.pipeline.surface(tube, color=edge_color)

    #mlab.close(fignum)
    #mlab.show() # interactive window


def read_ids_data(data_file, data_type='normal', labels_file='NSL_KDD/Field Names.csv'):
    # "Label" - "converter function" dictionary.
    label_dict = OrderedDict()
    result = []

    with open(labels_file) as lf:
        labels = csv.reader(lf)
        for label in labels:
            if len(label) == 1 or label[1] == 'continuous':
                label_dict[label[0]] = lambda l: np.float64(l)
            elif label[1] == 'symbolic':
                label_dict[label[0]] = lambda l: hash(l)

    f_list = [i for i in label_dict.values()]

    with open(data_file) as df:
        # data = csv.DictReader(df, label_dict.keys())
        data = csv.reader(df)
        for d in data:
            # Last two fields
            if d[-2] == data_type:
                result.append(tuple(f_list[n](i) for n, i in enumerate(d[:-2])))

    return result


class GNG():
    """Growing Neural Gas multidimensional implementation"""

    def __init__(self, data, surface_graph=None, eps_b=0.05, eps_n=0.0005, max_age=25,
                 lambda_=100, alpha=0.5, d=0.0005, max_nodes=100):
        """."""
        self.graph = nx.Graph()
        self.data = data
        self.eps_b = eps_b
        self.eps_n = eps_n
        self.max_age = max_age
        self.lambda_ = lambda_
        self.alpha = alpha
        self.d = d
        self.max_nodes = max_nodes
        self.num_of_input_signals = 0
        self._surface_graph = surface_graph

        self.pos = None

        node1 = data[np.random.randint(0, len(data))]
        node2 = data[np.random.randint(0, len(data))]

        # make sure you dont select same positions
        if self.is_nodes_equal(node1, node2):
            print("Rerun ---------------> similar nodes selected")
            return None

        # initialize here
        self.count = 0
        self.graph.add_node(self.count, pos=node1, error=0)
        self.count += 1
        self.graph.add_node(self.count, pos=node2, error=0)
        self.graph.add_edge(self.count - 1, self.count, age=0)

    def number_of_clusters(self):
        return nx.number_connected_components(self.graph)

    def is_nodes_equal(self, n1, n2):
        return len(set(n1) & set(n2)) == len(n1)

    def distance(self, a, b):
        """Calculate distance between two points."""

        # Euclidian distance.
        # return sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
        return euclidean(a, b)

    def determine_2closest_vertices(self, curnode):
        """Where this curnode is actually the x,y index of the data we want to analyze."""
        self.pos = nx.get_node_attributes(self.graph, 'pos')
        templist = []
        for node, position in iteritems(self.pos):
            dist = self.distance(curnode, position)
            templist.append([node, dist])

        distlist = np.array(templist)

        ind = np.lexsort((distlist[:, 0], distlist[:, 1]))
        distlist = distlist[ind]

        return distlist[0], distlist[1]

    def get_new_position(self, winnerpos, nodepos):
        """."""
        move_delta = tuple(self.eps_b * (i - j) for i, j in zip(nodepos, winnerpos))
        newpos = tuple(i + j for i, j in zip(move_delta, winnerpos))

        return newpos

    def get_new_position_neighbors(self, neighborpos, nodepos):
        """."""
        movement = tuple(self.eps_n * (i - j) for i, j in zip(nodepos, neighborpos))
        newpos = tuple(i + j for i, j in zip(neighborpos, movement))

        return newpos

    def update_winner(self, curnode):
        """."""
        # find nearest unit and second nearest unit
        winner1, winner2 = self.determine_2closest_vertices(curnode)
        winnernode = winner1[0]
        winnernode2 = winner2[0]
        win_dist_from_node = winner1[1]

        errorvectors = nx.get_node_attributes(self.graph, 'error')

        error1 = errorvectors[winner1[0]]
        # update the new error
        newerror = error1 + win_dist_from_node**2
        self.graph.add_node(winnernode, error=newerror)

        # move the winner node towards current node
        self.pos = nx.get_node_attributes(self.graph, 'pos')
        newposition = self.get_new_position(self.pos[winnernode], curnode)
        self.graph.add_node(winnernode, pos=newposition)

        # now update all the neighbors distances and their ages
        neighbors = nx.all_neighbors(self.graph, winnernode)
        age_of_edges = nx.get_edge_attributes(self.graph, 'age')
        for n in neighbors:
            newposition = self.get_new_position_neighbors(self.pos[n], curnode)
            self.graph.add_node(n, pos=newposition)
            key = (int(winnernode), n)
            if key in age_of_edges:
                newage = age_of_edges[(int(winnernode), n)] + 1
            else:
                newage = age_of_edges[(n, int(winnernode))] + 1
            self.graph.add_edge(winnernode, n, age=newage)

        # no sense in what I am writing here, but with algorithm it goes perfect
        # if winnner and 2nd winner are connected, update their age to zero
        if (self.graph.get_edge_data(winnernode, winnernode2) is not None):
            self.graph.add_edge(winnernode, winnernode2, age=0)
        else:
            # else create an edge between them
            self.graph.add_edge(winnernode, winnernode2, age=0)

        # if there are ages more than maximum allowed age, remove them
        age_of_edges = nx.get_edge_attributes(self.graph, 'age')
        for edge, age in iteritems(age_of_edges):

            if age > self.max_age:
                self.graph.remove_edge(edge[0], edge[1])

                # if it causes isolated vertix, remove that vertex as well

                for node in self.graph.nodes():
                    if not self.graph.neighbors(node):
                        self.graph.remove_node(node)

    def get_average_dist(self, a, b):
        """."""
        av_dist = tuple((i + j) / 2 for i, j in zip(a, b))

        return av_dist

    def save_img(self, fignum, output_images_dir='images'):
        """."""

        #fig = pl.figure(fignum)
        #ax = fig.add_subplot(111, projection='3d')
        if self._surface_graph is not None:
            #nx.draw(self._surface_graph, self._surface_pos, node_color='#ffffff', with_labels=False, node_size=100, alpha=0.5, width=1.5)
            draw_graph3d(self._surface_graph, fignum)
         
        #position = nx.get_node_attributes(self.graph, 'pos')
        draw_graph3d(self.graph, fignum, clear=False, node_color=(1, 0, 0))
        #nx.draw(self.graph, position, node_color='r', node_size=100, with_labels=False, edge_color='b', width=1.5, dim=3)
        #pl.title('Growing Neural Gas')
        mlab.savefig("{0}/{1}.png".format(output_images_dir, str(fignum)))

    def train(self, max_iterations=10000, output_images_dir='images'):
        """."""

        if os.path.isdir(output_images_dir):
            shutil.rmtree('{}'.format(output_images_dir))

        os.makedirs(output_images_dir)

        print("Ouput images will be saved in: {0}".format(output_images_dir))
        fignum = 0
        self.save_img(fignum, output_images_dir)

        for i in xrange(1, max_iterations):
            print("Iterating..{0:d}/{1}".format(i, max_iterations))
            for x in self.data:
                self.update_winner(x)

                # step 8: if number of input signals generated so far
                if i % self.lambda_ == 0 and len(self.graph.nodes()) <= self.max_nodes:
                    # find a node with the largest error
                    errorvectors = nx.get_node_attributes(self.graph, 'error')
                    import operator
                    node_largest_error = max(iteritems(errorvectors), key=operator.itemgetter(1))[0]

                    # find a node from neighbor of the node just found,
                    # with largest error
                    neighbors = self.graph.neighbors(node_largest_error)
                    max_error_neighbor = None
                    max_error = -1
                    errorvectors = nx.get_node_attributes(self.graph, 'error')
                    for n in neighbors:
                        if errorvectors[n] > max_error:
                            max_error = errorvectors[n]
                            max_error_neighbor = n

                    # insert a new unit half way between these two
                    self.pos = nx.get_node_attributes(self.graph, 'pos')

                    newnodepos = self.get_average_dist(self.pos[node_largest_error], self.pos[max_error_neighbor])
                    self.count = self.count + 1
                    newnode = self.count
                    self.graph.add_node(newnode, pos=newnodepos)

                    # insert edges between new node and other two nodes
                    self.graph.add_edge(newnode, max_error_neighbor, age=0)
                    self.graph.add_edge(newnode, node_largest_error, age=0)

                    # remove edge between the other two nodes

                    self.graph.remove_edge(max_error_neighbor, node_largest_error)

                    # decrease error variable of other two nodes by multiplying with alpha
                    errorvectors = nx.get_node_attributes(self.graph, 'error')
                    error_max_node = self.alpha * errorvectors[node_largest_error]
                    error_max_second = self.alpha * max_error
                    self.graph.add_node(max_error_neighbor, error=error_max_second)
                    self.graph.add_node(node_largest_error, error=error_max_node)

                    # initialize the error variable of newnode with max_node
                    self.graph.add_node(newnode, error=error_max_node)

                    fignum += 1
                    self.save_img(fignum, output_images_dir)

                # step 9: Decrease all error variables
                errorvectors = nx.get_node_attributes(self.graph, 'error')
                for i in self.graph.nodes():
                    olderror = errorvectors[i]
                    newerror = olderror - self.d * olderror
                    self.graph.add_node(i, error=newerror)


def sort_nicely(limages):
    """."""
    def convert(text): return int(text) if text.isdigit() else text

    def alphanum_key(key): return [convert(c) for c in re.split('([0-9]+)', key)]
    limages = sorted(limages, key=alphanum_key)
    return limages


def convert_images_to_gif(output_images_dir, output_gif):
    """Convert a list of images to a gif."""

    image_dir = "{0}/*.png".format(output_images_dir)
    list_images = glob.glob(image_dir)
    file_names = sort_nicely(list_images)
    images = [imageio.imread(fn) for fn in file_names]
    imageio.mimsave(output_gif, images)


def main():
    """."""

    #read_ids_data('NSL_KDD/Small Training Set.csv')
    #read_ids_data('NSL_KDD/20 Percent Training Set.csv')
    #read_ids_data('NSL_KDD/KDDTrain+.txt')

    G, pos = create_test_data_graph(read_test_file())
    #G, pos = create_data_graph(read_ids_data('NSL_KDD/Small Training Set.csv'))

    data = []
    for key, value in iteritems(pos):
        data.append([value[0], value[1], 0])

    data = np.array(data, dtype='float32')

    grng = GNG(data, surface_graph=G)

    output_images_dir = 'images'
    output_gif = 'output.gif'
    if grng is not None:
        grng.train(max_iterations=500)
        print('Clusters count: {}'.format(grng.number_of_clusters()))
        convert_images_to_gif(output_images_dir, output_gif)

    return 0


if __name__ == "__main__":
    exit(main())
