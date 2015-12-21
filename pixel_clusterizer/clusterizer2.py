import numpy as np
from numba import njit
import sys


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


@njit(cache=True)
def _reset_cluster_variables(actual_cluster_size, actual_cluster_id, max_cluster_charge):
    actual_cluster_size = 0
    actual_cluster_id = 0
    max_cluster_charge = 0


@njit(cache=True)
def _correct_cluster_id(hits, actual_event_number, actual_cluster_id, actual_event_hit_index_start, cluster_id):
    ' Substracts one from all cluster IDs of the event, starting from cluster id actual_cluster_id'
    for i in range(actual_event_hit_index_start, hits.shape[0]):
        if _new_event(hits[i].event_number, actual_event_number):  # Stop if new event is reached
            break
        if cluster_id[i] != -1 and cluster_id[i] > actual_cluster_id:
            cluster_id[i] -= 1

@njit(cache=True)
def _new_event(event_number, actual_event_number):
    'Detect a new event by checking if the event number of the actual hit is the actual event number'
    return event_number != actual_event_number

#@njit(cache=True)
def _merge_cluster(i, j, hits, is_seed, cluster_id, max_cluster_charge, max_cluster_id):
    is_seed[i] = 0  # Event hit is not necessarily seed anymore
    if hits[j].charge >= max_cluster_charge:  # Old cluster hit can be the seed, if charge is equal max_cluster_charge to keep lowest index max charge hit seed hit
        max_cluster_charge = hits[j].charge
    actual_cluster_id = cluster_id[j]  # Correct the actual cluster id
    cluster_id[i] = actual_cluster_id  # Actual hit belongs to other already existing cluster
    max_cluster_id = max_cluster_id - 1  # Actual new cluster ID is not used, since actual hit belongs to already existing cluster

#@njit
def cluster_hits(hits, cluster, x_cluster_distance=1, y_cluster_distance=1, frame_cluster_distance=4):
    cluster_id = np.zeros(shape=hits.shape, dtype=np.int16) - 1  # Cluster ID -1 means hit not assigned to cluster
    is_seed = np.zeros(shape=hits.shape, dtype=np.uint8)  # Seed 1 means hit is seed; lowest index hit with max charge hit is seed, thus there is always only one seed in a cluster
    cluster_size = np.zeros(shape=hits.shape, dtype=np.int16)  # Cluster size of the cluster the hit belongs to
    n_cluster = np.zeros(shape=hits.shape, dtype=np.int16)  # Number of clusters in the event the hit belongs to

    # Temporary variables that are reset for each cluster or event
    actual_event_number = 0
    actual_event_index_start = 0
    # actual_event_cluster_index_start = 0
    # actual_cluster_index = 0
    actual_event_hit_index_start = 0
    max_cluster_charge = 0
    actual_cluster_id = 0
    actual_cluster_size = 0
    max_cluster_id = 0
    actual_n_cluster = 0

    #print('Eventnumber', 'Column', 'Row', 'Cluster ID')

    # Outer loop over all hits in the array (refered to as actual hit)
    for i in range(hits.shape[0]):

        # Check for new event and reset event variables
        if _new_event(hits[i].event_number, actual_event_number):
            _reset_cluster_variables(actual_cluster_size, actual_cluster_id, max_cluster_charge)
            actual_event_hit_index_start = i
            # actual_event_cluster_index_start += n_cluster
            actual_event_number = hits[i].event_number
            max_cluster_id = 0
            actual_n_cluster = 0
            cluster_id[i] = actual_cluster_id  # First event hit is always asigned to cluster id = 0

        # Check if actual hit is already asigned to a cluster, if not define new actual cluster
        if cluster_id[i] == -1:  # Actual hit was never assigned to a cluster
            actual_cluster_size = 0
            actual_cluster_id = max_cluster_id  # Set actual cluster id
            max_cluster_id += 1  # Create new cluster ID that was not used before
            max_cluster_charge = hits[i].charge  # One hit with max_cluster_charge is seed
            is_seed[i] = 1  # First hit of cluster is seed until hiogher charge hit is found
            cluster_id[i] = actual_cluster_id  # Asign actual hit to actual cluster
            actual_n_cluster += 1  # Increase per event number of cluster count
            # actual_cluster_index = actual_event_cluster_index_start + n_cluster
        else:  # Hit was already asigned to a cluster in the inner loop, thus skip actual hit
            continue

        # Inner loop over actual event hits (refered to as event hit) and try to find hits belonging to the actual cluster
        for j in range(actual_event_hit_index_start, hits.shape[0]):

            # Stop hits of actual event loop if new event is reached
            if _new_event(hits[j].event_number, actual_event_number):
                break

            # Check if event hit belongs to actual hit
            if abs(int(hits[i].column) - int(hits[j].column)) <= x_cluster_distance and abs(int(hits[i].row) - int(hits[j].row)) <= y_cluster_distance and abs(int(hits[i].frame) - int(hits[j].frame)) <= frame_cluster_distance:

                # Check if event hit is already assigned to a acluster, can happen since the hits are not sorted
                if cluster_id[j] != -1 and cluster_id[j] < actual_cluster_id:  # event hit belongs already to a cluster A, thus actual hit is not a new cluster hit and actual hit also belongs to cluster A
                    _merge_cluster(i, j, hits, is_seed, cluster_id, max_cluster_charge, max_cluster_id)
#                     is_seed[i] = 0  # Event hit is not necessarily seed anymore
#                     if hits[j].charge >= max_cluster_charge:  # Old cluster hit can be the seed, if charge is equal max_cluster_charge to keep lowest index max charge hit seed hit
#                         max_cluster_charge = hits[j].charge
#                     actual_cluster_id = cluster_id[j]  # Correct the actual cluster id
#                     cluster_id[i] = actual_cluster_id  # Actual hit belongs to other already existing cluster
#                     max_cluster_id -= 1  # Actual new cluster ID is not used, since actual hit belongs to already existing cluster
                    print max_cluster_id
                    raise
                    _correct_cluster_id(hits, actual_event_number, actual_cluster_id, actual_event_hit_index_start, cluster_id)  # Make the cluster index increase by 1
                else:
                    actual_cluster_size += 1
                    cluster_id[j] = actual_cluster_id

                # Check if event hit has a higher charge, then make it the seed hit
                if hits[j].charge > max_cluster_charge:
                    is_seed[j] = 1
                    is_seed[i] = 0
                    max_cluster_charge = hits[j].charge

        cluster_size[i] = actual_cluster_size

        print i, cluster_id, is_seed

    return cluster_id, is_seed


if __name__ == '__main__':
    # create some fake data
    hits = np.ones(shape=(10, ), dtype=data_struct.HitInfo)
    cluster = np.zeros(shape=(hits.shape[0], ), dtype=data_struct.ClusterInfo)

    hits[0]['column'], hits[0]['row'], hits[0]['charge'], hits[0]['event_number'] = 0, 0, 30, 0
    hits[1]['column'], hits[1]['row'], hits[1]['charge'], hits[1]['event_number'] = 0, 2, 6, 0
    hits[2]['column'], hits[2]['row'], hits[2]['charge'], hits[2]['event_number'] = 0, 6, 30, 0
    hits[3]['column'], hits[3]['row'], hits[3]['charge'], hits[3]['event_number'] = 0, 4, 6, 0
    hits[4]['column'], hits[4]['row'], hits[4]['charge'], hits[4]['event_number'] = 0, 8, 6, 0
    hits[5]['column'], hits[5]['row'], hits[5]['charge'], hits[5]['event_number'] = 0, 1, 30, 0
    hits[6]['column'], hits[6]['row'], hits[6]['charge'], hits[6]['event_number'] = 0, 15, 6, 0
    hits[7]['column'], hits[7]['row'], hits[7]['charge'], hits[7]['event_number'] = 0, 14, 30, 0
    hits[8]['column'], hits[8]['row'], hits[8]['charge'], hits[8]['event_number'] = 0, 16, 6, 0
    hits[9]['column'], hits[9]['row'], hits[9]['charge'], hits[9]['event_number'] = 0, 13, 6, 0

    pprint_array(hits)

    hits_clustered = np.zeros(shape=hits.shape, dtype=data_struct.ClusterHitInfo)
    hits_clustered['column'] = hits['column']
    hits_clustered['row'] = hits['row']
    hits_clustered['charge'] = hits['charge']
    hits_clustered['event_number'] = hits['event_number']

    hits_clustered['cluster_ID'], hits_clustered['is_seed'] = cluster_hits(hits.view(np.recarray), cluster.view(np.recarray))

    pprint_array(hits_clustered)
