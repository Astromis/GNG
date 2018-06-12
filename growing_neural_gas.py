#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Initialize module utils."""

from math import sqrt
from mayavi import mlab
import operator
import imageio
from collections import OrderedDict
from scipy.spatial.distance import euclidean
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
import time


def sh(s):
    sum = 0
    for i, c in enumerate(s):
        sum += i * ord(c)
    return sum


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

    return preprocessing.normalize(result, axis=0, norm='max')


def draw_dots3d(dots, edges, fignum, clear=True,
                title = 'Incremental Growing Neuron Gas for the network anomalies detection',
                size=(1024, 768), graph_colormap='viridis',
                bgcolor=(1, 1, 1),
                node_color=(0.3, 0.65, 0.3), node_size=0.01,
                edge_color=(0.3, 0.3, 0.9), edge_size=0.003,
                text_size=0.14, text_color=(0, 0, 0), text_coords=[0.84, 0.75], text={},
                title_size=0.3,
                angle=get_ra()):

    # https://stackoverflow.com/questions/17751552/drawing-multiplex-graphs-with-networkx

    # numpy array of x, y, z positions in sorted node order
    xyz = shrink_to_3d(dots)

    if fignum == 0:
        mlab.figure(fignum, bgcolor=bgcolor, fgcolor=text_color, size=size)

    # Mayavi is buggy, and following code causes sockets leak.
    #if mlab.options.offscreen:
    #    mlab.figure(fignum, bgcolor=bgcolor, fgcolor=text_color, size=size)
    #elif fignum == 0:
    #    mlab.figure(fignum, bgcolor=bgcolor, fgcolor=text_color, size=size)

    if clear:
        mlab.clf()

    # the x,y, and z co-ordinates are here
    # manipulate them to obtain the desired projection perspective

    pts = mlab.points3d(xyz[:, 0], xyz[:, 1], xyz[:, 2],
                        scale_factor=node_size,
                        scale_mode='none',
                        color=node_color,
                        #colormap=graph_colormap,
                        resolution=20,
                        transparent=False)

    mlab.text(text_coords[0], text_coords[1], '\n'.join(['{} = {}'.format(n, v) for n, v in text.items()]), width=text_size)

    if clear:
        mlab.title(title, height=0.95)
        mlab.roll(next(angle))
        mlab.orientation_axes(pts)
        mlab.outline(pts)

    """
    for i, (x, y, z) in enumerate(xyz):
        label = mlab.text(x, y, str(i), z=z,
                          width=text_size, name=str(i), color=text_color)
        label.property.shadow = True
    """

    pts.mlab_source.dataset.lines = edges
    tube = mlab.pipeline.tube(pts, tube_radius=edge_size)
    mlab.pipeline.surface(tube, color=edge_color)

    #mlab.show() # interactive window


def draw_graph3d(graph, fignum, *args, **kwargs):
    graph_pos = nx.get_node_attributes(graph, 'pos')
    edges = np.array([e for e in graph.edges()])
    dots = np.array([graph_pos[v] for v in sorted(graph)], dtype='float64')

    draw_dots3d(dots, edges, fignum, *args, **kwargs)


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


def read_ids_data(data_file, activity_type='normal', labels_file='NSL_KDD/Field Names.csv', with_host=False):
    selected_parameters = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'serror_rate',
                           'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_srv_count', 'count']

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

    if activity_type == 'normal':
        data_type = lambda t: t == 'normal'
    elif activity_type == 'abnormal':
        data_type = lambda t: t != 'normal'
    elif activity_type == 'full':
        data_type = lambda t: True
    else:
        raise ValueError('`activity_type` must be "normal", "abnormal" or "full"')

    print('Reading {} activity from the file "{}" [generated host data {} included]...'.
          format(activity_type, data_file, 'was' if with_host else 'was not'))

    with open(data_file) as df:
        # data = csv.DictReader(df, label_dict.keys())
        data = csv.reader(df)
        for d in data:
            if data_type(d[-2]):
                # Skip last two fields and add only specified fields.
                net_params = tuple(f_list[n](i) for n, i in enumerate(d[:-2])) # if n_list[n] in selected_parameters)

                if with_host:
                    host_params = generate_host_activity(activity_type != 'abnormal')
                    result.append(net_params + host_params)
                else:
                    result.append(net_params)
    print('Records count: {}'.format(len(result)))
    return result


class IGNG():
    """Incremental Growing Neural Gas multidimensional implementation"""

    def __init__(self, data, surface_graph=None, eps_b=0.01, eps_n=0.002, max_age=10,
                 a_mature=1, output_images_dir='images'):
        """."""

        # Deviation parameters.
        self._dev_params = None
        self._graph = nx.Graph()
        self._data = data
        self._eps_b = eps_b
        self._eps_n = eps_n
        self._max_age = max_age
        self._a_mature = a_mature
        self._num_of_input_signals = 0
        self._surface_graph = surface_graph
        self._fignum = 0
        self._count = 0
        self._max_train_iters = 0
        self._start_time = time.time()

        # Initial value is a standard deviation of the data.
        self._d = np.std(data)

        self._output_images_dir = output_images_dir

        if os.path.isdir(output_images_dir):
            shutil.rmtree('{}'.format(output_images_dir))

        os.makedirs(output_images_dir)

        print("Ouput images will be saved in: {0}".format(output_images_dir))

    def number_of_clusters(self):
        return nx.number_connected_components(self._graph)

    def detect_anomalies(self, data, threshold=5, train=False, save_step=100):
        anomalies_counter, anomaly_records_counter, normal_records_counter = 0, 0, 0
        anomaly_level = 0

        start_time = self._start_time = time.time()

        for i, d in enumerate(data):
            risk_level = self.test_node(d, train)
            if risk_level != 0:
                anomaly_records_counter += 1
                anomaly_level += risk_level
                if anomaly_level > threshold:
                    anomalies_counter += 1
                    #print('Anomaly was detected [count = {}]!'.format(anomalies_counter))
                    anomaly_level = 0
            else:
                normal_records_counter += 1

            if i % save_step == 0:
                tm = time.time() - start_time
                print('Abnormal records = {}, Normal records = {}, Detection time = {} s, Time per record = {} s'.
                      format(anomaly_records_counter, normal_records_counter, round(tm, 2), tm / i if i else 0))

        tm = time.time() - start_time

        print('{} [abnormal records = {}, normal records = {}, detection time = {} s, time per record = {} s]'.
              format('Anomalies were detected (count = {})'.format(anomalies_counter) if anomalies_counter else 'Anomalies weren\'t detected',
                     anomaly_records_counter, normal_records_counter, round(tm, 2), tm / len(data)))

        return anomalies_counter > 0

    def test_node(self, node, train=False):
        dist = abs(self.__determine_closest_vertice_distance(node))
        dist_sub_dev = dist - self.__calculate_deviation_params()
        if dist_sub_dev > 0:
            return dist_sub_dev

        if dist > self.__calculate_deviation_params() and train:
            self.__train_on_data_item(node)

        return 0

    def train(self, max_iterations=100, save_step=0):
        """IGNG training method"""

        self._dev_params = None
        self._max_train_iters = max_iterations

        fignum = self._fignum
        self.__save_img(fignum, 0)
        CHS = self.__calinski_harabaz_score
        igng = self.__igng
        data = self._data

        if save_step < 1:
            save_step = max_iterations

        old = 0
        calin = CHS()
        i_count = 0
        start_time = self._start_time = time.time()

        while old - calin <= 0:
            print('Iteration {0:d}...'.format(i_count))
            i_count += 1
            steps = 1
            while steps <= max_iterations:
                for i, x in enumerate(data):
                    igng(x)
                    if i % save_step == 0:
                        tm = time.time() - start_time
                        print('Training time = {} s, Time per record = {} s, Training step = {}, Clusters count = {}, Neurons = {}, CHI = {}'.
                              format(round(tm, 2),
                                     tm * i_count * steps / (i if i else len(data)),
                                     i_count,
                                     self.number_of_clusters(),
                                     len(self._graph),
                                     old - calin)
                              )
                        self.__save_img(fignum, i_count)
                        fignum += 1
                steps += 1

            self._d -= 0.1 * self._d
            old = calin
            calin = CHS()
        print('Training complete, clusters count = {}'.format(self.number_of_clusters()))
        self._fignum = fignum

    def __fast_train_on_data_item(self, data_item):
        steps = 0
        while steps < max_iterations:
            igng(data_item)
            steps += 1

    def __train_on_data_item(self, data_item):
        """IGNG training method"""

        np.append(self._data, data_item)

        self._dev_params = None
        CHS = self.__calinski_harabaz_score
        igng = self.__igng
        data = self._data

        max_iterations = self._max_train_iters

        old = 0
        calin = CHS()
        i_count = 0

        # Strictly less.
        while old - calin < 0:
            print('Training with new normal node, step {0:d}...'.format(i_count))
            i_count += 1
            steps = 0

            if i_count > 100:
                print('BUG', old, calin)
                break

            while steps < max_iterations:
                igng(data_item)
                steps += 1
            self._d -= 0.1 * self._d
            old = calin
            calin = CHS()

    def __calculate_deviation_params(self, skip_embryo=True):
        if self._dev_params is not None:
            return self._dev_params

        dcvd = self.__determine_closest_vertice_distance
        dlen = len(self._data)
        dmean = np.mean(self._data, axis=0)
        deviation = 0

        for node in self._data:
            deviation += dcvd(node, skip_embryo)
        deviation /= dlen
        deviation = sqrt(deviation)
        self._dev_params = deviation

        return deviation

    def __calinski_harabaz_score(self, skip_embryo=True):
        graph = self._graph
        nodes = graph.nodes
        extra_disp, intra_disp = 0., 0.

        # CHI = [B / (c - 1)]/[W / (n - c)]
        # Total numb er of neurons.
        #ns = nx.get_node_attributes(self._graph, 'n_type')
        c = len([v for v in nodes.values() if v['n_type'] == 1]) if skip_embryo else len(nodes)
        # Total number of data.
        n = len(self._data)

        # Mean of the all data.
        mean = np.mean(self._data, axis=1)

        pos = nx.get_node_attributes(self._graph, 'pos')

        for node, k in pos.items():
            if skip_embryo and nodes[node]['n_type'] == 0:
                # Skip embryo neurons.
                continue
            
            mean_k = np.mean(k)
            extra_disp += len(k) * np.sum((mean_k - mean) ** 2)
            intra_disp += np.sum((k - mean_k) ** 2)

        return (1. if intra_disp == 0. else
                extra_disp * (n - c) /
                (intra_disp * (c - 1.)))

    def __determine_closest_vertice_distance(self, curnode, skip_embryo=True):
        """Where this curnode is actually the x,y index of the data we want to analyze."""

        pos = nx.get_node_attributes(self._graph, 'pos')
        nodes = self._graph.nodes

        distance = sys.maxint
        for node, position in pos.items():
            if skip_embryo and nodes[node]['n_type'] == 0:
                # Skip embryo neurons.
                continue
            dist = euclidean(curnode, position)
            if dist < distance:
                distance = dist
        return distance

    def __determine_2closest_vertices(self, curnode):
        """Where this curnode is actually the x,y index of the data we want to analyze."""

        pos = nx.get_node_attributes(self._graph, 'pos')

        winner1 = None
        winner2 = None

        for node, position in pos.items():
            dist = euclidean(curnode, position)
            if winner1 is None or dist < winner1[1]:
                winner1 = [node, dist]
                continue
            if winner2 is None or dist < winner2[1]:
                winner2 = [node, dist]

        return winner1, winner2

    def __get_specific_nodes(self, n_type):
        return [n for n, p in nx.get_node_attributes(self._graph, 'n_type').items() if p == n_type]

    def __igng(self, cur_node):
        """Main IGNG training subroutine"""

        # find nearest unit and second nearest unit
        winner1, winner2 = self.__determine_2closest_vertices(cur_node)
        graph = self._graph
        nodes = graph.nodes
        d = self._d

        # Second list element is a distance.
        if winner1 is None or winner1[1] >= d:
            # 0 - is an embryo type.
            graph.add_node(self._count, pos=cur_node, error=0, n_type=0, age=0)
            winner_node1 = self._count
            self._count += 1
            return
        else:
            winner_node1 = winner1[0]

        # Second list element is a distance.
        if winner2 is None or winner2[1] >= d:
            # 0 - is an embryo type.
            graph.add_node(self._count, pos=cur_node, error=0, n_type=0, age=0)
            winner_node2 = self._count
            self._count += 1
            graph.add_edge(winner_node1, winner_node2, age=0)
            return
        else:
            winner_node2 = winner2[0]

        # Increment the age of all edges, emanating from the winner.
        for e in graph.edges(winner_node1, data=True):
            e[2]['age'] += 1

        w_node = nodes[winner_node1]
        # Move the winner node towards current node.
        w_node['pos'] += self._eps_b * (cur_node - w_node['pos'])

        neighbors = nx.all_neighbors(graph, winner_node1)
        a_mature = self._a_mature

        for n in neighbors:
            c_node = nodes[n]
            # Move all direct neighbors of the winner.
            c_node['pos'] += self._eps_n * (cur_node - c_node['pos'])
            # Increment the age of all direct neighbors of the winner.
            c_node['age'] += 1
            if c_node['n_type'] == 0 and c_node['age'] >= a_mature:
                # Now, it's a mature neuron.
                c_node['n_type'] = 1

        # Create connection with age == 0 between two winners.
        graph.add_edge(winner_node1, winner_node2, age=0)

        max_age = self._max_age

        # If there are ages more than maximum allowed age, remove them.
        age_of_edges = nx.get_edge_attributes(graph, 'age')
        for edge, age in iteritems(age_of_edges):
            if age >= max_age:
                graph.remove_edge(edge[0], edge[1])

        # If it causes isolated vertix, remove that vertex as well.
        #graph.remove_nodes_from(nx.isolates(graph))
        for node, v in nodes.items():
            if v['n_type'] == 0:
                # Skip embryo neurons.
                continue
            if not graph.neighbors(node):
                graph.remove_node(node)

    def __save_img(self, fignum, training_step):
        """."""

        if self._surface_graph is not None:
            text = OrderedDict([
                ('Image', fignum),
                ('Training step', training_step),
                ('Time', '{} s'.format(round(time.time() - self._start_time, 2))),
                ('Clusters count', self.number_of_clusters()),
                ('Neurons', len(self._graph)),
                ('    Mature', len(self.__get_specific_nodes(1))),
                ('    Embryo', len(self.__get_specific_nodes(0))),
                ('Connections', len(self._graph.edges)),
                ('Data records', len(self._data))
            ])

            draw_graph3d(self._surface_graph, fignum)

        graph = self._graph

        if len(graph) > 0:
            #graph_pos = nx.get_node_attributes(graph, 'pos')
            #nodes = sorted(self.get_specific_nodes(1))
            #dots = np.array([graph_pos[v] for v in nodes], dtype='float64')
            #edges = np.array([e for e in graph.edges(nodes) if e[0] in nodes and e[1] in nodes])
            #draw_dots3d(dots, edges, fignum, clear=False, node_color=(1, 0, 0))

            draw_graph3d(graph, fignum, clear=False, node_color=(1, 0, 0), text=text)

        mlab.savefig("{0}/{1}.png".format(self._output_images_dir, str(fignum)))
        #mlab.close(fignum)


def sort_nicely(limages):
    """Numeric string sort"""
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


def test_detector(use_hosts_data, output_images_dir='images', output_gif = 'output.gif'):
    """Detector quality testing routine"""

    #data = read_ids_data('NSL_KDD/20 Percent Training Set.csv')
    frame = '-' * 70
    training_set = 'NSL_KDD/Small Training Set.csv'
    #training_set = 'NSL_KDD/KDDTest-21.txt'
    testing_set = 'NSL_KDD/KDDTest-21.txt'
    #testing_set = 'NSL_KDD/KDDTrain+.txt'

    print('{}\n{}\n{}'.format(frame, 'Detector training...', frame))
    data = read_ids_data(training_set, activity_type='normal')
    data = preprocessing.normalize(np.array(data, dtype='float64'), axis=0, norm='l2', copy=False)
    G = create_data_graph(data)

    gng = IGNG(data, surface_graph=G, output_images_dir=output_images_dir)
    gng.train(max_iterations=15, save_step=50)

    print('Saving GIF file...')
    convert_images_to_gif(output_images_dir, output_gif)

    print('{}\n{}\n{}'.format(frame, 'Apllying detector to the normal activity using the training set...', frame))
    gng.detect_anomalies(data)

    for a_type in ['abnormal', 'full']:
        print('{}\n{}\n{}'.format(frame, 'Apllying detector to the {} activity using the training set...'.format(a_type), frame))
        d_data = read_ids_data(training_set, activity_type=a_type)
        d_data = preprocessing.normalize(np.array(d_data, dtype='float64'), axis=0, norm='l2', copy=False)
        gng.detect_anomalies(d_data)

    dt = OrderedDict([('normal', None), ('abnormal', None), ('full', None)])

    for a_type in dt.keys():
        print('{}\n{}\n{}'.format(frame, 'Apllying detector to the {} activity using the testing set without adaptive learning...'.format(a_type), frame))
        d = read_ids_data(testing_set, activity_type=a_type)
        dt[a_type] = d = preprocessing.normalize(np.array(d, dtype='float64'), axis=0, norm='l2', copy=False)
        gng.detect_anomalies(d, save_step=1000)

    for a_type in ['full']:
        print('{}\n{}\n{}'.format(frame, 'Apllying detector to the {} activity using the testing set with adaptive learning...'.format(a_type), frame))
        gng.detect_anomalies(dt[a_type], train=True, save_step=1000)


def main():
    """Entry point"""

    start_time = time.time()

    mlab.options.offscreen = True
    test_detector(False)
    print('Full working time = {}'.format(round(time.time() - start_time, 2)))

    return 0


if __name__ == "__main__":
    exit(main())
