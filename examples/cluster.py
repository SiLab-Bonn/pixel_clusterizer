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
    # You can define your hit struct as you like; but it has to contain at least the field names defined in pixel_clusterizer.data_struct.HitInfo
    # The field data types do NOT have to be the same!
    HitStruct = np.dtype([('event_number', '<i8'),
                          ('frame', '<u1'),
                          ('column', '<u4'),
                          ('row', '<u4'),
                          ('charge', '<u1'),
                          ('parameter', '<i4')])

    # Create some fake data
    hits = np.ones(shape=(2, ), dtype=HitStruct)

    hits[0]['column'], hits[0]['row'], hits[0]['charge'], hits[0]['event_number'] = 17, 36, 30, 19
    hits[1]['column'], hits[1]['row'], hits[1]['charge'], hits[1]['event_number'] = 18, 36, 6, 19
#     hits[2]['column'], hits[2]['row'], hits[2]['charge'], hits[2]['event_number'] = 7, 7, 1, 19

    # create clusterizer object
    clusterizer = HitClusterizer()

    # all working settings are listed here with std. values
    clusterizer.set_x_cluster_distance(2)  # cluster distance in columns
    clusterizer.set_y_cluster_distance(2)  # cluster distance in rows
    clusterizer.set_frame_cluster_distance(4)   # cluster distance in time frames
    clusterizer.set_max_hit_charge(29)  # only add hits with charge <= 13

    # Main functions
    cluster_hits, cluster = clusterizer.cluster_hits(hits)  # cluster hits

    # print input / output histograms
    print('INPUT:')
    pprint_array(hits)
    print('OUTPUT:')
    print('Hits with cluster info:')
    pprint_array(cluster_hits)
    print('Cluster info:')
    pprint_array(cluster)
