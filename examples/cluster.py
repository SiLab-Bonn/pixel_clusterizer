''' Example how to use the clusterizer'''
import numpy as np
from builtins import str
import sys

from pixel_clusterizer.clusterizer import HitClusterizer
from pixel_clusterizer import data_struct


def pprint_array(array):  # just to print the results in a nice way
    offsets = []
    for column_name in array.dtype.names:
        sys.stdout.write(column_name)
        sys.stdout.write('\t')
        offsets.append(column_name.count(''))
    for row in array:
        print('')
        for i, column in enumerate(row):
            sys.stdout.write(' ' * (offsets[i] / 2))
            sys.stdout.write(str(column))
            sys.stdout.write('\t')
    print('')


if __name__ == "__main__":
    # create some fake data
    hits = np.ones(shape=(2, ), dtype=data_struct.HitInfo)

    hits[0]['column'], hits[0]['row'], hits[0]['charge'], hits[0]['event_number'] = 17, 36, 30, 19
    hits[1]['column'], hits[1]['row'], hits[1]['charge'], hits[1]['event_number'] = 18, 36, 6, 19
#     hits[2]['column'], hits[2]['row'], hits[2]['charge'], hits[2]['event_number'] = 7, 7, 1, 19

    # create clusterizer object
    clusterizer = HitClusterizer(n_columns=1000, n_rows=1000, n_frames=2, n_charges=31)

    # all working settings are listed here with std. values
    clusterizer.set_debug_output(False)
    clusterizer.set_info_output(False)
    clusterizer.set_warning_output(True)
    clusterizer.set_error_output(True)

    clusterizer.create_cluster_info_array(True)
    clusterizer.create_cluster_hit_info_array(True)  # std. setting is FALSE!

    clusterizer.set_x_cluster_distance(2)  # cluster distance in columns
    clusterizer.set_y_cluster_distance(2)  # cluster distance in rows
    clusterizer.set_frame_cluster_distance(4)   # cluster distance in time frames
    clusterizer.set_max_hit_charge(29)  # only add hits with charge <= 13

    # Cluster cuts
    clusterizer.set_max_cluster_hits(30)  # only add cluster with at most 30 hits
    clusterizer.set_min_cluster_hits(1)  # only add cluster with at least 1 hit

    # Main functions
    clusterizer.add_hits(hits)  # cluster hits
    cluster_hits = clusterizer.get_hit_cluster()
    cluster = clusterizer.get_cluster()

    # print input / output histograms
    print('INPUT:')
    pprint_array(hits)
    print('OUTPUT:')
    print('Hits with cluster info:')
    pprint_array(cluster_hits)
    print('Cluster info:')
    pprint_array(cluster)