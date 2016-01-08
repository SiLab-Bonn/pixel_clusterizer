import numpy as np
from numba import njit, jit
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
            #sys.stdout.write(' ' * (offsets[i] / 2))
            sys.stdout.write(str(column))
            sys.stdout.write('\t')
    print('')

@njit
def _correct_cluster_id(hits, actual_event_number, actual_cluster_id, actual_event_hit_index, cluster_id):
    ' Substracts one from all cluster IDs of the event, starting from cluster id actual_cluster_id'
    for i in range(actual_event_hit_index, hits.shape[0]):
        if _new_event(hits[i].event_number, actual_event_number):  # Stop if new event is reached
            break
        if cluster_id[i] != -1 and cluster_id[i] > actual_cluster_id:
            cluster_id[i] -= 1


@njit
def _new_event(event_number, actual_event_number):
    'Detect a new event by checking if the event number of the actual hit is the actual event number'
    return event_number != actual_event_number

@njit
def _merge_cluster(i, j, hits, cluster, is_seed, cluster_id, max_cluster_charge, next_cluster_id, actual_event_cluster_index):
    is_seed[i] = 0  # Event hit is not necessarily seed anymore
    if hits[j].charge >= max_cluster_charge:  # Old cluster hit can be the seed, if charge is equal max_cluster_charge to keep lowest index max charge hit seed hit
        max_cluster_charge = hits[j].charge
    actual_cluster_id = cluster_id[j]  # Correct the actual cluster id

    # Merge cluster infos
    cluster[actual_event_cluster_index + cluster_id[j]].n_hits += cluster[actual_event_cluster_index + cluster_id[i]].n_hits  # Add up cluster sizes of the merged cluster
    cluster[actual_event_cluster_index + actual_cluster_id].charge += cluster[actual_event_cluster_index + cluster_id[i]].charge  # Add up cluster charges of the merged cluster

    # Reset wrongly set new cluster value after merge
    cluster[actual_event_cluster_index + cluster_id[i]].n_hits = 0  # Reset wrongly set cluster value
    cluster[actual_event_cluster_index + cluster_id[i]].ID = 0  # Reset wrongly set cluster value
    cluster[actual_event_cluster_index + cluster_id[i]].charge = 0  # Reset wrongly set cluster value

    # Correct cluster IDs of the hit / cluster array
    cluster[actual_event_cluster_index + actual_cluster_id].ID = actual_cluster_id  # Set ID of actual cluster
    cluster_id[i] = actual_cluster_id  # Actual hit belongs to other already existing cluster

    next_cluster_id -= 1  # Actual new cluster ID is not used, since actual hit belongs to already existing cluster

    return max_cluster_charge, next_cluster_id, actual_cluster_id

@njit
def _finish_event(hits, cluster, is_seed, n_cluster, cluster_size, cluster_id, actual_event_hit_index, new_actual_event_hit_index, next_cluster_id, actual_event_cluster_index):
    ''' Set hit and cluster information of the last finished event (like number of cluster in this event (n_cluster),  cluster charge ...). '''
    for i in range(actual_event_hit_index, new_actual_event_hit_index):
        actual_cluster_id = cluster_id[i]  # Set cluster ID from the cluster the actual hit belongs to

        # Set hit cluster info
        n_cluster[i] = next_cluster_id
        cluster_size[i] = cluster[cluster_id[i]].n_hits

        # Set cluster info from hits
        if is_seed[i] == 1:  # Set seed hit cluster info
            cluster[actual_event_cluster_index + actual_cluster_id].seed_column = hits[i].column
            cluster[actual_event_cluster_index + actual_cluster_id].seed_row = hits[i].row
        # Add cluster position as sum of all hit positions weighted by the charge (center of gravity)
        # the position is in the center of the pixel (column = 0 == mean_column = 0.5)
        cluster[actual_event_cluster_index + actual_cluster_id].mean_column += (hits[i].column + 0.5) * hits[i].charge
        cluster[actual_event_cluster_index + actual_cluster_id].mean_row += (hits[i].row + 0.5) * hits[i].charge

    # Normalize cluster position by the charge for center of gravity
    for i in range(actual_event_cluster_index, actual_event_cluster_index + next_cluster_id):
        if cluster[i].charge > 0:
            cluster[i].mean_column /= cluster[i].charge
            cluster[i].mean_row /= cluster[i].charge

@njit
def cluster_hits(hits, cluster, x_cluster_distance=1, y_cluster_distance=1, frame_cluster_distance=4):
    # Additional cluster info for the hit array
    cluster_id = np.zeros(shape=hits.shape, dtype=np.int16) - 1  # Cluster ID -1 means hit not assigned to cluster
    is_seed = np.zeros(shape=hits.shape, dtype=np.uint8)  # Seed 1 means hit is seed; lowest index hit with max charge hit is seed, thus there is always only one seed in a cluster
    cluster_size = np.zeros(shape=hits.shape, dtype=np.int16)  # Cluster size of the cluster the hit belongs to
    n_cluster = np.zeros(shape=hits.shape, dtype=np.int16)  # Number of clusters in the event the hit belongs to

    # Temporary variables that are reset for each cluster or event
    actual_event_number, actual_event_hit_index, actual_event_cluster_index, actual_cluster_id, max_cluster_charge, next_cluster_id = 0, 0, 0, 0, 0, 0

    # Outer loop over all hits in the array (refered to as actual hit)
    for i in range(hits.shape[0]):

        # Check for new event and reset event variables
        if _new_event(hits[i].event_number, actual_event_number):
            print(0)
            _finish_event(hits, cluster, is_seed, n_cluster, cluster_size, cluster_id, actual_event_hit_index, i, next_cluster_id, actual_event_cluster_index)
            actual_event_hit_index = i
            actual_event_cluster_index = actual_event_cluster_index + next_cluster_id
            actual_event_number = hits[i].event_number
            #print(next_cluster_id)
            next_cluster_id = 0  # First cluster has ID 1

        # Check if actual hit is already asigned to a cluster, if not define new actual cluster containing with the actual hit as the first hit
        if cluster_id[i] == -1:  # Actual hit was never assigned to a cluster
            print(1)
            actual_cluster_id = next_cluster_id  # Set actual cluster id
            next_cluster_id += 1  # Create new cluster ID that was not used before
            max_cluster_charge = hits[i].charge  # One hit with max_cluster_charge is seed
            is_seed[i] = 1  # First hit of cluster is seed until hiogher charge hit is found
            cluster_id[i] = actual_cluster_id  # Assign actual hit to actual cluster

            # Set new cluster data
            cluster[actual_event_cluster_index + actual_cluster_id].charge = max_cluster_charge  # Set charge of the cluster to first hit charge
            cluster[actual_event_cluster_index + actual_cluster_id].ID = actual_cluster_id  # Set the cluster id
            cluster[actual_event_cluster_index + actual_cluster_id].event_number = actual_event_number  # Set the event number
            cluster[actual_event_cluster_index + actual_cluster_id].n_hits = 1  # Increase the size counter of the actual cluster
        else:  # Hit was already assigned to a cluster in the inner loop, thus skip actual hit
            continue

        # Inner loop over actual event hits (refered to as event hit) and try to find hits belonging to the actual cluster
        for j in range(actual_event_hit_index, hits.shape[0]):

            # Omit if event hit is actual hit (clustering with itself)
            if j == i:
                continue

            # Stop event hits loop if new event is reached
            if _new_event(hits[j].event_number, actual_event_number):
                break

            # Check if event hit belongs to actual hit and thus to the actual cluster
            if abs(int(hits[i].column) - int(hits[j].column)) <= x_cluster_distance and abs(int(hits[i].row) - int(hits[j].row)) <= y_cluster_distance and abs(int(hits[i].frame) - int(hits[j].frame)) <= frame_cluster_distance:

                # Check if event hit is already assigned to a acluster, can happen since the hits are not sorted
                if cluster_id[j] != -1 and cluster_id[j] < actual_cluster_id:  # Event hit belongs already to a cluster A, thus actual hit is not a new cluster hit and actual hit also belongs to cluster A
                    max_cluster_charge, next_cluster_id, actual_cluster_id = _merge_cluster(i, j, hits, cluster, is_seed, cluster_id, max_cluster_charge, next_cluster_id, actual_event_cluster_index)  # Merge the new cluster to the already existing old
                    _correct_cluster_id(hits, actual_event_number, actual_cluster_id, actual_event_hit_index, cluster_id)  # Correct the cluster indices to make them increase by 1
                else:
                    cluster_id[j] = actual_cluster_id  # Add event hit to actual cluster
                    cluster[actual_event_cluster_index + actual_cluster_id].n_hits += 1  # Increaser the number of hits of the actual cluster
                    cluster[actual_event_cluster_index + actual_cluster_id].charge += hits[j].charge  # Add charge to the actual cluster charge

                # Check if event hit has a higher charge, then make it the seed hit
                if hits[j].charge > max_cluster_charge:
                    # Event hit is seed and not actual hit, thus switch the seed flag
                    is_seed[j] = 1
                    is_seed[i] = 0

                    # Event hit defines the highest hit charge of the actual cluster
                    max_cluster_charge = hits[j].charge

        # cluster_size[i] = actual_cluster_size

        # print(i, cluster_id, is_seed)

    # Last event is assumed to be finished at the end of the hit array, thus add info
    _finish_event(hits, cluster, is_seed, n_cluster, cluster_size, cluster_id, actual_event_hit_index, i + 1, next_cluster_id, actual_event_cluster_index)
    return cluster_id, is_seed, cluster_size, n_cluster, actual_event_cluster_index + next_cluster_id


if __name__ == '__main__':
    # create some fake data
    hits = np.ones(shape=(20, ), dtype=data_struct.HitInfo)
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

    hits[10]['column'], hits[10]['row'], hits[10]['charge'], hits[10]['event_number'] = 0, 0, 30, 1
    hits[11]['column'], hits[11]['row'], hits[11]['charge'], hits[11]['event_number'] = 0, 2, 6, 1
    hits[12]['column'], hits[12]['row'], hits[12]['charge'], hits[12]['event_number'] = 0, 6, 30, 1
    hits[13]['column'], hits[13]['row'], hits[13]['charge'], hits[13]['event_number'] = 0, 4, 6, 1
    hits[14]['column'], hits[14]['row'], hits[14]['charge'], hits[14]['event_number'] = 0, 8, 6, 1
    hits[15]['column'], hits[15]['row'], hits[15]['charge'], hits[15]['event_number'] = 0, 1, 30, 1
    hits[16]['column'], hits[16]['row'], hits[16]['charge'], hits[16]['event_number'] = 0, 15, 6, 1
    hits[17]['column'], hits[17]['row'], hits[17]['charge'], hits[17]['event_number'] = 0, 14, 30, 1
    hits[18]['column'], hits[18]['row'], hits[18]['charge'], hits[18]['event_number'] = 0, 16, 6, 1
    hits[19]['column'], hits[19]['row'], hits[19]['charge'], hits[19]['event_number'] = 0, 13, 6, 1

# create some fake data
#     hits = np.ones(shape=(5, ), dtype=data_struct.HitInfo)
#     cluster = np.zeros(shape=(hits.shape[0], ), dtype=data_struct.ClusterInfo)
#  
#     hits[0]['column'], hits[0]['row'], hits[0]['charge'], hits[0]['event_number'] = 0, 0, 30, 0
#     hits[1]['column'], hits[1]['row'], hits[1]['charge'], hits[1]['event_number'] = 0, 1, 6, 0
#     hits[2]['column'], hits[2]['row'], hits[2]['charge'], hits[2]['event_number'] = 0, 2, 30, 0
#     hits[3]['column'], hits[3]['row'], hits[3]['charge'], hits[3]['event_number'] = 0, 3, 6, 0
#     hits[4]['column'], hits[4]['row'], hits[4]['charge'], hits[4]['event_number'] = 0, 8, 6, 0

    pprint_array(hits)

    hits_clustered = np.zeros(shape=hits.shape, dtype=data_struct.ClusterHitInfo)
    hits_clustered['column'] = hits['column']
    hits_clustered['row'] = hits['row']
    hits_clustered['charge'] = hits['charge']
    hits_clustered['event_number'] = hits['event_number']

    hits_clustered['cluster_ID'], hits_clustered['is_seed'], hits_clustered['cluster_size'], hits_clustered['n_cluster'], n_cluster = cluster_hits(hits.view(np.recarray), cluster.view(np.recarray))

    pprint_array(hits_clustered)
    pprint_array(cluster[:n_cluster])
