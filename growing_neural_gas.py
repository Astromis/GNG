# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""Initialize module utils."""

from mayavi import mlab
import operator
import imageio
from collections import OrderedDict
from scipy.spatial.distance import euclidean
from sklearn.metrics import calinski_harabaz_score
from sklearn import preprocessing
import csv
import numpy as np
import networkx as nx
import re
import os
import shutil
import sys
import glob
from past.builtins import xrange
from future.utils import iteritems


def sh(s):
    sum = 0
    for i, c in enumerate(s):
        sum += i * ord(c)
    return sum


def read_test_file():
    """Read the file and return the indices as list of lists."""
    filename = 'test_data.txt'
    with open(filename) as file:
        array2d = [[int(digit) for digit in line.split()] for line in file]

    return array2d


def create_test_data_graph(array2d):
    """Create the graph and returns the networkx version of it 'G'."""

    row, column = len(array2d), len(array2d[0])
    count = 0

    G = nx.Graph()

    for j in xrange(column):
        for i in xrange(row):
            if array2d[row - 1 - i][j] == 0:
                G.add_node(count, pos=(j, i, np.random.randint(-1, 1)))
                count += 1

    """
    for index in pos.keys():
        for index2 in pos.keys():
            if pos[index][0] == pos[index2][0] and pos[index][1] == pos[index2][1] - 1:
                G.add_edge(index, index2, weight=1)
            if pos[index][1] == pos[index2][1] and pos[index][0] == pos[index2][0] - 1:
                G.add_edge(index, index2, weight=1)
    """
    return G


def create_data_graph(dots):
    """Create the graph and returns the networkx version of it 'G'."""

    count = 0

    G = nx.Graph()

    for i in dots:
        G.add_node(count, pos=(i))
        count += 1

    return G


def get_ra(ra=0, ra_step=0.3):
    while True:
        if ra >= 360:
            ra = 0
        else:
            ra += ra_step
        yield ra


def shrink_to_3d(data):
    result = []

    for i in data:
        depth = len(i)
        if depth <= 3:
            result.append(i)
        else:
            sm = np.sum([(n) * v for n, v in enumerate(i[2:])])
            if sm == 0:
                sm = 1

            r = np.array([i[0], i[1], i[2]])
            r *= sm
            r /= np.sum(r)

            result.append(r)

    return preprocessing.normalize(result)


def draw_graph3d(graph, fignum, clear=True,
                 size=(1024, 768), graph_colormap='viridis',
                 bgcolor = (1, 1, 1),
                 node_color=(0.3, 0.65, 0.3), node_size=0.01,
                 edge_color=(0.3, 0.3, 0.9), edge_size=0.003,
                 text_size=0.008, text_color=(0, 0, 0),
                 title_size=0.3,
                 angle=get_ra()):

    # https://stackoverflow.com/questions/17751552/drawing-multiplex-graphs-with-networkx

    gr = graph #nx.convert_node_labels_to_integers(graph)
    graph_pos = nx.get_node_attributes(graph, 'pos')
    
    # numpy array of x, y, z positions in sorted node order
    xyz = shrink_to_3d(np.array([graph_pos[v] for v in sorted(gr)], dtype='float32'))

    # scalar colors
    #scalars = np.array([n for n in gr.nodes()])
    
    if mlab.options.offscreen:
        mlab.figure(fignum, bgcolor=bgcolor, fgcolor=text_color, size=size)
    else:
        if fignum == 0:
            mlab.figure(fignum, bgcolor=bgcolor, fgcolor=text_color, size=size)

    if clear:
        mlab.clf()

    # the x,y, and z co-ordinates are here
    # manipulate them to obtain the desired projection perspective 

    pts = mlab.points3d(xyz[:, 0], xyz[:, 1], xyz[:, 2],
                        # scalars,
                        scale_factor=node_size,
                        scale_mode='none',
                        color=node_color,
                        #colormap=graph_colormap,
                        resolution=20,
                        transparent=False)

    if clear:
        mlab.title('Growing Neuron Gas for the network anomalies detection', height=0.95)
        mlab.roll(next(angle))
        mlab.orientation_axes(pts)
        mlab.outline(pts)

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


def generate_host_activity(is_normal):
    # Host loads is changed only in 25% cases.
    attack_percent = 25
    up_level = (20, 30)

    # CPU load in percent.
    cpu_load = (10, 30)
    # Disk IO per second.
    iops = (10, 50)
    # Memory consumption in percent.
    mem_cons = (30, 60)

    cur_up_level = 0

    if not is_normal and np.random.randint(0, 100) < attack_percent:
        cur_up_level = np.random.randint(*up_level)

    cpu_load = np.random.randint(cur_up_level + cpu_load[0], cur_up_level + cpu_load[1])
    iops = np.random.randint(cur_up_level + iops[0], cur_up_level + iops[1])
    mem_cons = np.random.randint(cur_up_level + mem_cons[0], cur_up_level + mem_cons[1])

    return cpu_load, iops, mem_cons


def read_ids_data(data_file, is_normal=True, labels_file='NSL_KDD/Field Names.csv', with_host=False):
    selected_parameters = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent']

    # "Label" - "converter function" dictionary.    
    label_dict = OrderedDict()
    result = []

    with open(labels_file) as lf:
        labels = csv.reader(lf)
        for label in labels:
            if len(label) == 1 or label[1] == 'continuous':
                label_dict[label[0]] = lambda l: np.float64(l)
            elif label[1] == 'symbolic':
                label_dict[label[0]] = lambda l: sh(l)

    f_list = [i for i in label_dict.values()]
    n_list = [i for i in label_dict.keys()]

    data_type = lambda t: t == 'normal' if is_normal else t != 'normal'

    with open(data_file) as df:
        # data = csv.DictReader(df, label_dict.keys())
        data = csv.reader(df)
        for d in data:
            if data_type(d[-2]):
                # Skip last two fields and add only specified fields.
                net_params = tuple(f_list[n](i) for n, i in enumerate(d[:-2]) if n_list[n] in selected_parameters)

                if with_host:
                    host_params = generate_host_activity(is_normal)
                    result.append(net_params + host_params)
                else:
                    result.append(net_params)

    return result


class IGNG():
    """Incremental Growing Neural Gas multidimensional implementation"""

    def __init__(self, data, surface_graph=None, eps_b=0.2, eps_n=0.006, max_age=50,
                 lambda_=100, a_mature=1, d=0.995, max_nodes=100,
                 output_images_dir='images'):
        """."""
        self.graph = nx.Graph()
        self.data = data
        self.eps_b = eps_b
        self.eps_n = eps_n
        self.max_age = max_age
        self.lambda_ = lambda_
        self.a_mature = a_mature
        self.d = d
        self.max_nodes = max_nodes
        self.num_of_input_signals = 0
        self._stat = {
            'nodes': 0, # removed or added nodes per iter.
            'edges': 0, # removed or added edges per iter.
        }
        self._surface_graph = surface_graph

        self._fignum = 0

        # initialize here
        self.count = 0
        
        self._output_images_dir = output_images_dir

        if os.path.isdir(output_images_dir):
            shutil.rmtree('{}'.format(output_images_dir))

        os.makedirs(output_images_dir)

        print("Ouput images will be saved in: {0}".format(output_images_dir))

    def calinski_harabaz_score(self):
        return
        #calinski_harabaz_score(self.data, np.array([graph_pos[v] for v in sorted(self.graph)], dtype='float32'))
        extra_disp, intra_disp = 0., 0.

        # CHI = [B / (c - 1)]/[W / (n - c)]
        # Total numb er of neurons.
        c = self.count
        # Total number of data.
        n = len(self.data)
        
        # Mean of the all data.
        mean = np.mean(self.data, axis=0)
        
        # Neuron reference vector.
        nrv = np.array([graph_pos[v] for v in sorted(self.graph)], dtype='float32')
        
        print(mean, c, n, nrv)
        print(mean - nrv)
        exit(0)

        for k in range(n_labels):
            cluster_k = X[labels == k]
            mean_k = np.mean(cluster_k, axis=0)
            extra_disp += len(cluster_k) * np.sum((mean_k - mean) ** 2)
            intra_disp += np.sum((cluster_k - mean_k) ** 2)

        return (1. if intra_disp == 0. else
                extra_disp * (n_samples - n_labels) /
                (intra_disp * (n_labels - 1.)))

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
        #python growing_neural_gas.py  320,47s user 1,33s system 99% cpu 5:22,95 total
        pos = nx.get_node_attributes(self.graph, 'pos')

        winner1 = None
        winner2 = None

        for node, position in iteritems(pos):
            dist = self.distance(curnode, position)
            if winner1 is None or dist < winner1[1]:
                winner1 = [node, dist]
                continue
            if winner2 is None or dist < winner2[1]:
                winner2 = [node, dist]

        return winner1, winner2

    def update_winner(self, cur_node):
        """."""

        # find nearest unit and second nearest unit
        winner1, winner2 = self.determine_2closest_vertices(cur_node)
        graph = self.graph

        if winner1 is None or winner1[1] >= self.d:
            # 0 - is an embryo type.
            graph.add_node(self.count, pos=cur_node, error=0, n_type=0, age=0)
            winner_node1 = self.count
            self.count += 1
            return
        else:
            winner_node1 = winner1[0]

        if winner2 is None or winner2[1] >= self.d:
            # 0 - is an embryo type.
            graph.add_node(self.count, pos=cur_node, error=0, n_type=0, age=0)
            winner_node2 = self.count
            self.count += 1
            graph.add_edge(winner_node1, winner_node2, age=0)
            return
        else:
            winner_node2 = winner2[0]

        # Increment the age of all edges, emanating from the winner.
        for e in graph.edges(winner_node1, data=True):
            e[2]['age'] += 1

        w_node = graph.nodes[winner_node1]
        # Move the winner node towards current node.
        w_node['pos'] += self.eps_b * (cur_node - w_node['pos']) #self.get_new_position(w_node['pos'], cur_node)

        neighbors = nx.all_neighbors(graph, winner_node1)
        a_mature = self.a_mature

        for n in neighbors:
            c_node = graph.nodes[n]
            # Move all direct neighbors of the winner.
            c_node['pos'] += self.eps_n * (cur_node - c_node['pos'])
            # Increment the age of all direct neighbors of the winner.
            c_node['age'] += 1
            if c_node['n_type'] == 0 and c_node['age'] >= a_mature:
                # Now, it's a mature neuron.
                c_node['n_type'] = 1

        # Create connection with age == 0 between two winners.
        graph.add_edge(winner_node1, winner_node2, age=0)

        max_age = self.max_age

        # If there are ages more than maximum allowed age, remove them.
        age_of_edges = nx.get_edge_attributes(graph, 'age')
        for edge, age in iteritems(age_of_edges):
            if age >= max_age:
                #!!!
                graph.remove_edge(edge[0], edge[1])

        # if it causes isolated vertix, remove that vertex as well
        for node in graph.nodes():
            if not graph.neighbors(node):
                #!!!
                graph.remove_node(node)

    def get_average_dist(self, a, b):
        """."""
        av_dist = tuple((i + j) / 2 for i, j in zip(a, b))

        return av_dist

    def save_img(self, fignum):
        """."""

        if self._surface_graph is not None:
            draw_graph3d(self._surface_graph, fignum)
        
        if not fignum:
            return

        draw_graph3d(self.graph, fignum, clear=False, node_color=(1, 0, 0))
        mlab.savefig("{0}/{1}.png".format(self._output_images_dir, str(fignum)))

    def train(self, max_iterations=100):
        """."""

        fignum = self._fignum
        self.save_img(fignum)

        for i in xrange(1, max_iterations):
            print('CHS', self.calinski_harabaz_score())
            print('Iterating..{0:d}/{1}'.format(i, max_iterations))
            for x in self.data:
                self.update_winner(x)
            self.d -= 0.1 * self.d
            fignum += 1
            self.save_img(fignum)

        self._fignum = fignum

    def test_node(self, node):
        winner1, winner2 = self.determine_2closest_vertices(node)
        winnernode = winner1[0]
        winnernode2 = winner2[0]
        win_dist_from_node = winner1[1]
        graph = self.graph

        errorvectors = nx.get_node_attributes(self.graph, 'error')

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

    mlab.options.offscreen = True
    output_images_dir = 'images'

    #G = create_test_data_graph(read_test_file())
    data = read_ids_data('NSL_KDD/Small Training Set.csv')
    #data = read_ids_data('NSL_KDD/20 Percent Training Set.csv')
    #data = read_ids_data('NSL_KDD/KDDTrain+.txt')
    data = preprocessing.scale(preprocessing.normalize(np.array(data, dtype='float32'), copy=False), with_mean=False, copy=False)
    G = create_data_graph(data)

    #G = create_data_graph(read_ids_data('NSL_KDD/20 Percent Training Set.csv'))
    #data = []
    #for key, value in iteritems(pos):
    #    data.append(value)

    gng = IGNG(data, surface_graph=G, output_images_dir=output_images_dir)

    output_gif = 'output.gif'
    if gng is not None:
        gng.train(max_iterations=20)
        print('Clusters count: {}'.format(gng.number_of_clusters()))
        convert_images_to_gif(output_images_dir, output_gif)

    return 0


if __name__ == "__main__":
    exit(main())
