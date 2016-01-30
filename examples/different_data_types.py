''' This example shows howto setup the clusterizer to accept your own hit definition (different names and data types)'''
import numpy as np
from builtins import str
import sys

from pixel_clusterizer.clusterizer import HitClusterizer


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
    # A custom hit structure is defined here with unique names and data types
    hit_dtype = np.dtype([('timestamp', '<f8'),
                          ('time_window', '<u1'),
                          ('x', '<u4'),
                          ('y', '<u4'),
                          ('tot', '<u1')])

    # A mapping to the internal hit field has to be done
    hit_fields = {'timestamp': 'event_number',
                  'x': 'column',
                  'y': 'row',
                  'tot': 'charge',
                  'time_window': 'frame'
                  }

    # Create some fake data
    hits = np.ones(shape=(3, ), dtype=hit_dtype)
    hits[0]['x'], hits[0]['y'], hits[0]['tot'], hits[0]['timestamp'] = 17, 36, 7, 1.0
    hits[1]['x'], hits[1]['y'], hits[1]['tot'], hits[1]['timestamp'] = 18, 36, 6, 1.0
    hits[2]['x'], hits[2]['y'], hits[2]['tot'], hits[2]['timestamp'] = 7, 7, 1, 1.1

    # Initialize clusterizer object
    clusterizer = HitClusterizer(hit_fields=hit_fields, hit_dtype=hit_dtype)

    # Main function
    cluster_hits, cluster = clusterizer.cluster_hits(hits)  # cluster hits

    # Print input / output histograms
    print('INPUT:')
    pprint_array(hits)
    print('OUTPUT:')
    print('Hits with cluster info:')
    pprint_array(cluster_hits)
    print('Cluster info:')
    pprint_array(cluster)
