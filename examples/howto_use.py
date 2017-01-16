''' Example how to use the clusterizer'''
import numpy as np
from builtins import str
import sys

from pixel_clusterizer import clusterizer


def pprint_array(array):  # Just to print the arrays in a nice way
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
    # You can define your hit struct as you like; but it has to either contain the field names defined in pixel_clusterizer.data_struct.HitInfo or a mapping of the names have to be provided.
    # The field data types do NOT have to be the same!
    hit_dtype = np.dtype([('event_number', '<i8'),
                          ('frame', '<u1'),
                          ('column', '<u4'),
                          ('row', '<u4'),
                          ('charge', '<u1'),
                          ('custom_parameter', '<i4')])

    # Create some fake data
    hits = np.ones(shape=(3, ), dtype=hit_dtype)
    hits[0]['column'], hits[0]['row'], hits[0]['charge'], hits[0]['event_number'] = 17, 36, 11, 19
    hits[1]['column'], hits[1]['row'], hits[1]['charge'], hits[1]['event_number'] = 18, 36, 6, 19
    hits[2]['column'], hits[2]['row'], hits[2]['charge'], hits[2]['event_number'] = 7, 7, 1, 19

    # Initialize clusterizer object
    clusterizer = clusterizer.HitClusterizer()

    # All cluster settings are listed here with their std. values
    clusterizer.set_column_cluster_distance(2)  # cluster distance in columns
    clusterizer.set_row_cluster_distance(2)  # cluster distance in rows
    clusterizer.set_frame_cluster_distance(4)   # cluster distance in time frames
    clusterizer.set_max_hit_charge(13)  # only add hits with charge <= 29
    clusterizer.ignore_same_hits(True)  # Ignore same hits in an event for clustering
    clusterizer.set_hit_dtype(hit_dtype)  # Set the data type of the hits (parameter data types and names)
    clusterizer.set_hit_fields({'event_number': 'event_number',  # Set the mapping of the hit names to the internal names (here there is no mapping done, this is the std. setting)
                                'column': 'column',
                                'row': 'row',
                                'charge': 'charge',
                                'frame': 'frame'
                                })

    # Main functions
    cluster_hits, clusters = clusterizer.cluster_hits(hits)  # cluster hits

    # Print input / output histograms
    print('INPUT:')
    pprint_array(hits)
    print('OUTPUT:')
    print('Hits with cluster info:')
    pprint_array(cluster_hits)
    print('Cluster info:')
    pprint_array(clusters)
