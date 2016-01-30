import numpy as np
from numba import njit, jit
import sys
import logging

from pixel_clusterizer import data_struct
from docutils.utils.roman import OutOfRangeError


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


# Fast functions that are compiled in time via numba
@njit()
def _new_event(event_number, actual_event_number):
    'Detect a new event by checking if the event number of the actual hit is the actual event number'
    return event_number != actual_event_number


@njit()
def _finish_event(hits, cluster, is_seed, n_cluster, cluster_size, cluster_id, actual_event_hit_index, new_actual_event_hit_index, next_cluster_id, actual_event_cluster_index):
    ''' Set hit and cluster information of the last finished event (like number of cluster in this event (n_cluster),  cluster charge ...). '''
    for i in range(actual_event_hit_index, new_actual_event_hit_index):  # Set hit cluster info that is only known at the end of the event
        n_cluster[i] = next_cluster_id

    # Normalize cluster position by the charge for center of gravity
    for i in range(actual_event_cluster_index, actual_event_cluster_index + next_cluster_id):
        cluster[i].mean_column /= (cluster[i].charge + cluster[i].n_hits)
        cluster[i].mean_row /= (cluster[i].charge + cluster[i].n_hits)


@njit()
def _reset_cluster_hit_indices(actual_cluster_hit_indices, actual_cluster_size):
    ''' Sets the cluster hit indices to the std. valie -1. To be able to use this array in the new event'''
    for i in range(actual_cluster_size):
        actual_cluster_hit_indices[i] = -1


@njit()
def _is_in_max_difference(value_1, value_2, max_difference):
    ''' Helper function to determine the difference of two values that can be np.uints. Works in python and numba mode.
    Circumvents numba bug #1653'''
    if value_1 <= value_2:
        return value_2 - value_1 <= max_difference
    return value_1 - value_2 <= max_difference


@njit()
def cluster_hits(hits, cluster, n_hits, x_cluster_distance=1, y_cluster_distance=2, frame_cluster_distance=4, max_n_cluster_hits=30, max_cluster_hit_charge=13, ignore_same_hits=True):
    ''' Main precompiled function that loopes over the hits and clusters them '''
    # Additional cluster info for the hit array
    cluster_id = np.zeros(shape=hits.shape, dtype=np.int16) - 1  # Cluster ID -1 means hit not assigned to cluster
    is_seed = np.zeros(shape=hits.shape, dtype=np.uint8)  # Seed 1 means hit is seed; lowest index hit with max charge hit is seed, thus there is always only one seed in a cluster
    cluster_size = np.zeros(shape=hits.shape, dtype=np.int16)  # Cluster size of the cluster the hit belongs to
    n_cluster = np.zeros(shape=hits.shape, dtype=np.int16)  # Number of clusters in the event the hit belongs to

    # Temporary variables that are reset for each cluster or event
    actual_event_number, actual_event_hit_index, actual_event_cluster_index, actual_cluster_id, max_cluster_charge, next_cluster_id, actual_cluster_size, actual_cluster_hit_index, seed_index = 0, 0, 0, 0, 0, 0, 0, 0, 0
    actual_cluster_hit_indices = np.zeros(shape=max_n_cluster_hits, dtype=np.int16) - 1  # The hit indices of the actual cluster, -1 means not assigned

    # Outer loop over all hits in the array (refered to as actual hit)
    for i in range(hits.shape[0]):
        if i >= n_hits:
            break

        # Check for new event and reset event variables
        if _new_event(hits[i].event_number, actual_event_number):
            _finish_event(hits, cluster, is_seed, n_cluster, cluster_size, cluster_id, actual_event_hit_index, i + 1, next_cluster_id, actual_event_cluster_index)
            actual_event_hit_index = i
            actual_event_cluster_index = actual_event_cluster_index + next_cluster_id
            actual_event_number = hits[i].event_number
            next_cluster_id = 0  # First cluster has ID 1

        # Reset temp array with hit indices of actual cluster for the next cluster
        _reset_cluster_hit_indices(actual_cluster_hit_indices, actual_cluster_size)
        actual_cluster_hit_index = 0

        # Check if actual hit is already asigned to a cluster, if not define new actual cluster containing the actual hit as the first hit
        if cluster_id[i] != -1:  # Hit was already assigned to a cluster in the inner loop, thus skip actual hit
            continue

        # Omit hits with charge > max_cluster_hit_charge
        if hits[i].charge > max_cluster_hit_charge:
            continue

        # Set/reset cluster variables for new cluster
        actual_cluster_size = 1  # actual cluster has one hit so far
        actual_cluster_id = next_cluster_id  # Set actual cluster id
        next_cluster_id += 1  # Create new cluster ID that was not used before
        max_cluster_charge = hits[i].charge  # One hit with max_cluster_charge is seed
        actual_cluster_hit_indices[actual_cluster_hit_index] = i - actual_event_hit_index

        # Set first hit cluster hit info
        is_seed[i] = 1  # First hit of cluster is seed until higher charge hit is found
        seed_index = i
        cluster_id[i] = actual_cluster_id  # Assign actual hit to actual cluster

        # Set cluster info from first hit
        cluster[actual_event_cluster_index + actual_cluster_id].event_number = actual_event_number
        cluster[actual_event_cluster_index + actual_cluster_id].ID = actual_cluster_id
        cluster[actual_event_cluster_index + actual_cluster_id].n_hits = 1
        cluster[actual_event_cluster_index + actual_cluster_id].charge = hits[i].charge
        cluster[actual_event_cluster_index + actual_cluster_id].mean_column += (hits[i].column + 0.5) * (hits[i].charge + 1)
        cluster[actual_event_cluster_index + actual_cluster_id].mean_row += (hits[i].row + 0.5) * (hits[i].charge + 1)
        cluster[actual_event_cluster_index + actual_cluster_id].seed_column = hits[i].column
        cluster[actual_event_cluster_index + actual_cluster_id].seed_row = hits[i].row

        for j in actual_cluster_hit_indices:  # Loop over all hits of the actual cluster; actual_cluster_hit_indices is updated within the loop if new hit are found
            if j == -1:  # There are no more cluster hits found
                break

            actual_inner_loop_hit_index = j + actual_event_hit_index

            # Inner loop over actual event hits (refered to as event hit) and try to find hits belonging to the actual cluster
            for k in range(i + 1, hits.shape[0]):
                # Omit if event hit is already belonging to a cluster
                if cluster_id[k] != -1:
                    continue

                if k >= n_hits:
                    break

                if hits[k].charge > max_cluster_hit_charge:
                    continue

                # Omit if event hit is actual hit (clustering with itself)
                if k == actual_inner_loop_hit_index:
                    continue

                # Stop event hits loop if new event is reached
                if _new_event(hits[k].event_number, actual_event_number):
                    break

                # Check if event hit belongs to actual hit and thus to the actual cluster
                if _is_in_max_difference(hits[actual_inner_loop_hit_index].column, hits[k].column, x_cluster_distance) and _is_in_max_difference(hits[actual_inner_loop_hit_index].row, hits[k].row, y_cluster_distance) and _is_in_max_difference(hits[actual_inner_loop_hit_index].frame, hits[k].frame, frame_cluster_distance):
                    if not ignore_same_hits or hits[actual_inner_loop_hit_index].column != hits[k].column or hits[actual_inner_loop_hit_index].row != hits[k].row:
                        actual_cluster_size += 1
                        actual_cluster_hit_index += 1
                        if actual_cluster_hit_index >= max_n_cluster_hits:
                            raise OutOfRangeError('There is a cluster with more than the specified max_cluster_hits. Increase this parameter!')
                        actual_cluster_hit_indices[actual_cluster_hit_index] = k - actual_event_hit_index
                        cluster_id[k] = actual_cluster_id  # Add event hit to actual cluster

                        # Add cluster position as sum of all hit positions weighted by the charge (center of gravity)
                        # the position is in the center of the pixel (column = 0 == mean_column = 0.5)
                        cluster[actual_event_cluster_index + actual_cluster_id].mean_column += (hits[k].column + 0.5) * (hits[k].charge + 1)
                        cluster[actual_event_cluster_index + actual_cluster_id].mean_row += (hits[k].row + 0.5) * (hits[k].charge + 1)
                        cluster[actual_event_cluster_index + actual_cluster_id].n_hits += 1
                        cluster[actual_event_cluster_index + actual_cluster_id].charge += hits[k].charge

                        # Check if event hit has a higher charge, then make it the seed hit
                        if hits[k].charge > max_cluster_charge:
                            # Event hit is seed and not actual hit, thus switch the seed flag
                            is_seed[k] = 1
                            is_seed[seed_index] = 0
                            seed_index = k
                            max_cluster_charge = hits[k].charge
                            # Set new seed hit in the cluster
                            cluster[actual_event_cluster_index + actual_cluster_id].seed_column = hits[k].column
                            cluster[actual_event_cluster_index + actual_cluster_id].seed_row = hits[k].row
                    else:
                        cluster_id[k] = -2  # Mark a ignored hit with index = -2

        # Set cluster size info for actual cluster hits
        for j in actual_cluster_hit_indices:  # Loop over all hits of the actual cluster; actual_cluster_hit_indices is updated within the loop if new hit are found
            if j == -1:  # there are no more cluster hits found
                break
            cluster_size[j + actual_event_hit_index] = actual_cluster_size

    # Last event is assumed to be finished at the end of the hit array, thus add info
    _finish_event(hits, cluster, is_seed, n_cluster, cluster_size, cluster_id, actual_event_hit_index, i + 1, next_cluster_id, actual_event_cluster_index)
    return cluster_id, is_seed, cluster_size, n_cluster, actual_event_cluster_index + next_cluster_id


class HitClusterizer(object):
    ''' Clusterizer class providing an interface for the jitted functions and storing settings.'''
    def __init__(self, max_hits=10000):
        # Std. settings
        self._create_cluster_hit_info_array = False
        self._max_cluster_hit_charge = 13
        self._max_hits = max_hits
        self._x_cluster_distance = 1
        self._y_cluster_distance = 2
        self._frame_cluster_distance = 4
        self._max_cluster_hits = 300
        self._ignore_same_hits = True

        self.hits_clustered = np.zeros(shape=(self._max_hits, ), dtype=data_struct.ClusterHitInfo)
        self.cluster = np.zeros(shape=(self._max_hits, ), dtype=data_struct.ClusterInfo).view(np.recarray)  # Only recarrays no structured arrays are supported by numba

        self.n_cluster = 0
        self.n_hits = 0

    def reset(self):
        raise NotImplementedError

    def set_max_hit_charge(self, value):
        self._max_cluster_hit_charge = value

    def set_x_cluster_distance(self, value):
        self._x_cluster_distance = value

    def set_y_cluster_distance(self, value):
        self._y_cluster_distance = value

    def set_frame_cluster_distance(self, value):
        self._frame_cluster_distance = value

    def set_max_cluster_hits(self, value):
        self._max_cluster_hits = value

    def ignore_same_hits(self, value):  # Ignore same hit in an event for clustering
        self._ignore_same_hits = value

    def create_cluster_hit_info_array(self, value=True):  # TODO: do not create cluster hit info of false to save time
        self._create_cluster_hit_info_array = value

    def cluster_hits(self, hits):
        self.n_hits = 0  # Effectively deletes the already clustered hits
        self._delete_cluster()  # Delete the already created cluster
        # The hit info is extended by the cluster info; this is only possible by creating a new hit info array
        self.hits_clustered['frame'][self.n_hits:hits.shape[0]] = hits['frame']
        self.hits_clustered['column'][self.n_hits:hits.shape[0]] = hits['column']
        self.hits_clustered['row'][self.n_hits:hits.shape[0]] = hits['row']
        self.hits_clustered['charge'][self.n_hits:hits.shape[0]] = hits['charge']
        self.hits_clustered['event_number'][self.n_hits:hits.shape[0]] = hits['event_number']

        self.hits_clustered['cluster_ID'][self.n_hits:hits.shape[0]], self.hits_clustered['is_seed'][self.n_hits:hits.shape[0]], self.hits_clustered['cluster_size'][self.n_hits:hits.shape[0]], self.hits_clustered['n_cluster'][self.n_hits:hits.shape[0]], self.n_cluster = cluster_hits(hits.view(np.recarray), self.cluster, n_hits=hits.shape[0], x_cluster_distance=self._x_cluster_distance, y_cluster_distance=self._y_cluster_distance, frame_cluster_distance=self._frame_cluster_distance, max_n_cluster_hits=self._max_cluster_hits, max_cluster_hit_charge=self._max_cluster_hit_charge, ignore_same_hits=self._ignore_same_hits)

        self.n_hits += hits.shape[0]

        return self.hits_clustered[:self.n_hits], self.cluster[:self.n_cluster]

    def get_hit_cluster(self):
        hits_clustered = self.hits_clustered[:self.n_hits]
        self.n_hits = 0
        return hits_clustered

    def get_cluster(self):
        cluster = self.cluster[:self.n_cluster]
        self._delete_cluster()
        return cluster

    def _delete_cluster(self):
        self.cluster = np.zeros(shape=(self._max_hits, ), dtype=data_struct.ClusterInfo).view(np.recarray)  # Only recarrays no structured arrays are supported by numba
        self.n_cluster = 0

