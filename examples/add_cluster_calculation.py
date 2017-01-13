''' This example shows how to set a end_of_cluster_functions that creates an additional cluster result. Here the seed charge of the cluster is additionally set.'''
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
    hit_dtype = np.dtype([('event_number', '<i8'),
                          ('frame', '<u1'),
                          ('column', '<u2'),
                          ('row', '<u2'),
                          ('charge', '<u2')])

    # Create some fake data
    hits = np.ones(shape=(3, ), dtype=hit_dtype)
    hits[0]['column'], hits[0]['row'], hits[0]['charge'], hits[0]['event_number'] = 17, 36, 11, 19
    hits[1]['column'], hits[1]['row'], hits[1]['charge'], hits[1]['event_number'] = 18, 36, 6, 19
    hits[2]['column'], hits[2]['row'], hits[2]['charge'], hits[2]['event_number'] = 7, 7, 1, 19

    # Initialize clusterizer object
    clusterizer = HitClusterizer()
    clusterizer.add_cluster_field(description=('seed_charge', '<u1'))  # Add an additional field to hold the charge of the seed hit

    # The end of loop function has to define all of the following arguments, even when they are not used
    # It has to be compile able by numba in non python mode
    # This end_of_cluster_function sets the additional seed_charge field
    def end_of_cluster_function(hits, clusters, cluster_size, cluster_hit_indices, cluster_index, cluster_id, charge_correction, noisy_pixels, disabled_pixels, seed_hit_index):
        clusters[cluster_index].seed_charge = hits[seed_hit_index].charge

    clusterizer.set_end_of_cluster_function(end_of_cluster_function)  # Set the new function to the clusterizer

    # Main function
    cluster_hits, clusters = clusterizer.cluster_hits(hits)  # cluster hits

    # Print input / output histograms
    print('INPUT:')
    pprint_array(hits)
    print('OUTPUT:')
    print('Hits with cluster info:')
    pprint_array(cluster_hits)
    print('Cluster info:')
    pprint_array(clusters)
